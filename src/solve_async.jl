import SciMLBase
using OrdinaryDiffEq
using Distributed
using Logging

const SIGNAL_WORKER = 1
const SIGNAL_CONTROL = 0
const SIGNAL_DONE = -1

"""
    solve_async_worker(prob, alg, interval; <keyword arguments>)

Perform fine integration of problem `prob` with integration scheme `alg` on interval `interval`.
This function should not be called directly, but is calledfrom [`solve_async`](@ref)
on a separate worker or thread.
"""
function solve_async_worker(
    prob::ODEProblem,
    alg,
    interval::Int;
    maxit::Int,
    info_channel::Union{Channel,RemoteChannel},
    data_channel::Union{Channel,RemoteChannel},
    t_0::Float64,
    t_end::Float64,
    kwargs...)

    ############################################################################
    ############################   PARAREAL SETUP   ############################
    ############################################################################

    # set up integrator
    int = init(prob, alg;
        tstops=[t_end],
        advance_to_tstop=true,
        kwargs...)

    ############################################################################
    ##########################   PARAREAL ITERATION   ##########################
    ############################################################################
    iteration = 0
    limit = min(maxit, interval)
    while iteration < limit
        # wait for signal from control node
        signal = fetch(info_channel)

        if signal == SIGNAL_DONE
            break
        elseif signal == SIGNAL_WORKER
            # increment counter
            iteration += 1

            # receive initial value from control node
            initial_value = take!(data_channel)

            # fine integration
            # don't reset stats here - total work counted at the end
            reinit!(int, initial_value, t0=t_0)
            step!(int)

            # write result to coarse channel. Used for update equation
            put!(data_channel, int.sol.u[end])

            # clear signal to transfer control
            take!(info_channel)
        else
            @warn "Worker $(id()) received unexpected signal"
            sleep(1e-3)
        end
    end
    return int.sol
end


"""
    solve_async(prob, alg; <keyword arguments>)

Solve given ODEProblem `prob` with Parareal, using the integration scheme `alg`.
Parallel ("fine") integration is delegated to other workers or threads,
sequential ("coarse") integration is performed
on the worker / thread this function is called on.

If `shared_memory` is set to `false` (default), parallel integration
is performed using the `Distributed` framework's worker model,
where workers are added via `addprocs()` and may live on different machines.
This can generate some communication overhead, as the values
at synchronization points need to be sent over the network.

Delegation to particular workers is achieved by passing their IDs
in `worker_ids`. If none are supplied, work may be delegated to all available workers.

WARNING: To use the distributed implementation, it has to be loaded
on all workers. After adding workers, run `@everywhere using Parareal`.
If your problem makes use of global variables, make sure they're available
to all workers by prefixing their declaration with `@everywhere`.

If `shared_memory` is set to `true`, parallel integration is performed using `Threads`.
This requires Julia to be started with the appropriate number of threads,
e.g. by calling `julia -t N` from the command line. See [https://docs.julialang.org/en/v1/manual/multi-threading/#Starting-Julia-with-multiple-threads]
for other options. The shared memory implementation is expected to be faster for
some problems, as the communication overhead is reduced. Allocation-heavy code, however,
can be severely bottlenecked by Julia's non-concurrent GC implementation. In these cases,
the `Distributed` implementation is expected to be faster.

# Arguments
- `parareal_intervals::Integer`: The number of intervals used in the Parareal algorithm.
- `maxit`: Maximum number of Parareal iterations to perform. Will be set to `parareal_intervals` if not provided.
- `norm::Function`: The norm by which to judge the discontinuities at interval interfaces. Default: Maximum absolute difference, `(x, y) -> maximum(abs.(x-y))`.
- `reltol::Float`: Relative tolerance on Parareal error as determined by `norm`.
- `abstol::Float`: Absolute tolerance on Parareal error as determined by `norm`.
- `coarse_args`: Keyword-arguments to be used in sequential ("coarse") integration.
- `fine_args`: Keyword-arguments to be used in parallel ("fine") integration.
- `init_args`: Keyword-arguments to be used in the initial sequential integration. If none are provided, `coarse_args` are used.
- Additional keyword-arguments are passed to all solvers. In case of keyword conflicts, the values specified in `coarse_args`, `fine_args` or `init_args` are preferred.

See also: [`solve_async_worker`](@ref), [`solve_sync`](@ref).
"""
function solve_async(
    prob::ODEProblem, alg;
    shared_memory=false,
    worker_ids=workers(),
    statistics=true,
    parareal_intervals::Int,
    norm=(x, y) -> maximum(abs.(x - y)),
    reltol=1e-3::Float64,
    abstol=1e-6::Float64,
    maxit=parareal_intervals::Int,
    coarse_args=(;),
    fine_args=(;),
    init_args=(;),
    kwargs...
)

    ############################################################################
    ###########################   SETUP DISTRIBUTED   ##########################
    ############################################################################

    # set up info channel to each worker. Signals what to do with values in data_channel.
    info_channels = [
        shared_memory ?
        Channel{Int}(1) :
        RemoteChannel(() -> Channel{Int}(1), myid())
        for _ = 1:parareal_intervals]

    # set up channel for coarse result for each worker.
    data_channels = [
        shared_memory ?
        Channel{typeof(prob.u0)}(1) :
        RemoteChannel(() -> Channel{typeof(prob.u0)}(1), myid())
        for _ = 1:parareal_intervals]

    # vector of Futures for the function calls to workers.
    # Each worker will return its full solution via this Future
    worker_futures = shared_memory ?
                     Vector{Task}(undef, parareal_intervals) :
                     Vector{Future}(undef, parareal_intervals)


    ############################################################################
    ############################   SETUP PARAREAL   ############################
    ############################################################################
    sync_points = LinRange(
        prob.tspan[1], prob.tspan[end], parareal_intervals + 1)

    # solution values at sync points. These are modified in the
    # parareal update eqn. and used as initial values in subsequent iterations.
    sync_values = Vector{typeof(prob.u0)}(undef, parareal_intervals)
    sync_values[1] = prob.u0

    # results of coarse propagator at sync points from prev. iteration.
    # saved to avoid re-computing.
    # coarse_prev[i] contains value at sync_points[i+1]
    # obtained from coarse propagation started at sync_points[i]
    coarse_prev = Vector{typeof(prob.u0)}(undef, parareal_intervals - 1)

    # errors at sync points. Difference between left and right solution,
    # measured by given norm
    # sync_errors[k] holds the error at the interface between intervals k and k+1
    sync_errors_abs = fill(Inf, parareal_intervals - 1)
    sync_errors_rel = fill(Inf, parareal_intervals - 1)

    # statistics
    if statistics
        stats_total = SciMLBase.DEStats()
    end

    if parareal_intervals > 1
        coarse_int = init(prob, alg;
            kwargs...,
            coarse_args...,
            # step to sync points
            tstops=sync_points,
            advance_to_tstop=true,
            # only save result at sync points
            saveat=sync_points,
        )

        # only construct initial integrator if separate options are given
        if !isempty(init_args)
            initial_int = init(prob, alg;
                kwargs...,
                init_args...,
                # step to sync points
                tstops=sync_points,
                advance_to_tstop=true,
                # only save result at sync points
                saveat=sync_points,
            )
        else
            initial_int = coarse_int
        end
    end

    # TODO if no workers were added, workers() = [1], but we should still use threads
    if shared_memory == false && parareal_intervals > 1 && length(worker_ids) > 0
        # prepare cachingpool
        cpool = CachingPool(worker_ids)
    else
        cpool = nothing
    end

    ############################################################################
    ############################   START WORKERS    ############################
    ############################################################################
    for interval = 1:parareal_intervals
        # function call on worker
        fn = solve_async_worker

        fn_args = (prob, alg, interval,)
        fn_kwargs = (;
            maxit=maxit,
            info_channel=info_channels[interval],
            data_channel=data_channels[interval],
            t_0=sync_points[interval],
            t_end=sync_points[interval+1],
            kwargs...,
            fine_args...,
        )

        worker_count = length(worker_ids)

        if shared_memory || worker_count == 0
            worker_futures[interval] = Threads.@spawn fn(fn_args...; fn_kwargs...)
        else
            worker_futures[interval] = remotecall(fn, cpool, fn_args...; fn_kwargs...)
            #worker_futures[interval] = @spawnat worker_ids[((interval-1)%worker_count)+1] fn()
        end
    end

    ############################################################################
    ############################   INITIALIZATION    ###########################
    ############################################################################

    # handle first interval
    put!(data_channels[1], sync_values[1])
    put!(info_channels[1], SIGNAL_WORKER)

    if parareal_intervals > 1
        reinit!(initial_int, sync_values[1], t0=sync_points[1])

        @debug "Starting sequential initialization"
        for interval = 1:parareal_intervals-1
            # initial integration
            step!(initial_int)
            coarse_result = initial_int.sol.u[end]

            # push result to worker
            put!(data_channels[interval+1], coarse_result)
            put!(info_channels[interval+1], SIGNAL_WORKER)

            sync_values[interval+1] = coarse_result
            coarse_prev[interval] = coarse_result
        end

        # add statistics from initialization
        if statistics
            _add_stats!(stats_total, initial_int.sol.stats)
        end
        @debug "Finished sequential initialization"
    end

    # asynchronous task: merge full results
    # depending on parareal settings, some to most workers finish early.
    # transferring full solution and merging it to the overall solution
    # can be done while the control node is waiting for other workers.
    merge = Threads.@spawn begin
        merged_sol = fetch(worker_futures[1])
        if statistics
            _add_stats!(stats_total, merged_sol.stats)
        end

        for interval = 2:parareal_intervals
            sol = fetch(worker_futures[interval])
            if statistics
                _add_stats!(stats_total, sol.stats)
            end
            merge_solution!(merged_sol, sol)
        end

        if statistics
            _reset_stats!(merged_sol.stats)
            _add_stats!(merged_sol.stats, stats_total)
        end
        return merged_sol
    end

    # asynchronous task: observe workers for errors, throw error
    error = Threads.@spawn begin
        while true
            for fut in worker_futures
                if istaskdone(fut)
                    fetch(fut)
                end
            end
            sleep(1e-3)
        end
    end


    ############################################################################
    ##############################   MAIN LOOP    ##############################
    ############################################################################

    retcode = :Default
    iteration = 1
    maxit = min(maxit, parareal_intervals)

    while iteration <= maxit
        @debug "Iteration $iteration in main Parareal loop"

        for interval = iteration:parareal_intervals
            @debug "Starting sequential update for interval $interval in iteration $iteration"

            # make sure fine result is available
            wait_for_empty(info_channels[interval], worker_futures[iteration])

            # get result
            fine_result = take!(data_channels[interval])

            # first active interval is exact now.
            if interval == iteration && interval > 1
                # parareal exactness
                @debug "Setting synchronization errors to zero for interface between intervals $(interval-1) and $(interval)"
                sync_errors_abs[interval-1] = sync_errors_rel[interval-1] = 0
            end

            # if in last interval: Break after setting sync errors
            if interval == parareal_intervals
                break
            else
                # set sync error to the right of current interval
                @debug "Computing synchronization errors for interface between intervals $(interval) and $(interval+1)"
                sync_errors_abs[interval] = norm(sync_values[interval+1], fine_result)
                sync_errors_rel[interval] = sync_errors_abs[interval] / norm(zeros(size(fine_result)), fine_result)

                # no coarse integration, no update equation
                if interval == iteration
                    @debug "Set sync value at interface between intervals $(interval) and $(interval+1) to fine result"
                    sync_values[interval+1] = fine_result
                    # tell worker to stop
                    put!(info_channels[interval], SIGNAL_DONE)
                    continue
                end

                if interval < parareal_intervals
                    @debug "Starting coarse integration over interval $interval in iteration $iteration"
                    # reseting the integrator does not reset the statistics!
                    statistics && _reset_stats!(coarse_int.sol.stats)
                    # coarse integration
                    reinit!(coarse_int, sync_values[interval], t0=sync_points[interval])
                    step!(coarse_int)
                    coarse_result = coarse_int.sol.u[end]

                    # update equation
                    sync_values[interval+1] = coarse_result + fine_result - coarse_prev[interval]

                    # set coarse_prev for next iteration
                    coarse_prev[interval] = coarse_result

                    # add statistics
                    statistics && _add_stats!(stats_total, coarse_int.sol.stats)
                end
            end
        end

        if maximum(sync_errors_abs, init=0.0) <= abstol || maximum(sync_errors_rel, init=0.0) <= reltol
            @debug "Parareal synchronization errors below tolerance"
            retcode = :Success
            break
        end

        # if max. iterations reached, break here. Sequential update not needed
        if iteration == maxit
            @debug "Max. iterations reached"
            retcode = :MaxIters
            break
        end

        # update for next iteration
        for interval = iteration+1:parareal_intervals
            # make sure we can write
            wait_for_empty(info_channels[interval], worker_futures[iteration])

            # write data
            put!(data_channels[interval], sync_values[interval])
            put!(info_channels[interval], SIGNAL_WORKER)
        end

        iteration += 1
    end

    ############################################################################
    ##########################   COMBINE SOLUTIONS   ###########################
    ############################################################################

    # tell workers we are done
    for interval = 1:parareal_intervals
        ch = info_channels[interval]

        # already marked as done
        if isready(ch) && fetch(ch) == SIGNAL_DONE
            continue
        end

        # if still running, wait until signal channel is empty
        wait_for_empty(ch)
        # then tell worker to quit
        put!(ch, SIGNAL_DONE)
    end
    @debug "Shut down remaining workers"

    # wait
    sol = fetch(merge)
    @debug "Async result collection finished"
    !isnothing(cpool) && clear!(cpool)

    # collect info
    info = (;
        retcode=retcode,
        iterations=iteration,
        abs_error=maximum(sync_errors_abs, init=0.0),
        rel_error=maximum(sync_errors_rel, init=0.0),
    )

    return sol, info
end
