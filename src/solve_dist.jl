import SciMLBase
using DifferentialEquations
using Distributed

const SIGNAL_WORKER = 1
const SIGNAL_CONTROL = 0
const SIGNAL_DONE = -1

"""
    Helper function called on each worker.
    Receives updated initial values from control node,
    performs integration and sends end value back to control node.
"""
function solve_dist_worker(
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
            # clear signal
            take!(info_channel)

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
            put!(info_channel, SIGNAL_CONTROL)
            continue
        elseif signal == SIGNAL_CONTROL
            sleep(1e-3)
            continue
        else
            sleep(1e-3)
            continue
        end
    end
    return int.sol
end


"""
    Main function called on control node.
    Performs sequential integration, waits for workers to finish parallel integration, then applies update formula.

    If `shared_memory` is set to `true` (default), parallel integration is
    performed using thread-parallel workers (on the same machine). This
    approach requires at least `parareal_intervals` processsors for optimal
    performance.
    Otherwise, parallel integration is performed using distributed workers
    (separate julia processes which may be on different machines). This
    approach does not exploit shared memory and is thus expected to be slower
    when run on a single machine.
"""
function solve_dist(
    prob::ODEProblem, alg;
    shared_memory=true,
    parareal_intervals::Int,
    tol=1e-3::Float64,
    norm=(x, y) -> maximum(abs.(x - y)),
    maxit=parareal_intervals::Int,
    coarse_args=(),
    fine_args=(),
    init_args=coarse_args
)

    ############################################################################
    ###########################   SETUP DISTRIBUTED   ##########################
    ############################################################################
    if !shared_memory
        worker_num = nworkers()
        worker_map = Dict{Int,Int}()
        for i = 1:parareal_intervals
            worker_map[i] = workers()[(i-1)%worker_num+1]
        end
    end


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
    sync_errors = fill(Inf, parareal_intervals - 1)

    # statistics
    stats_total = SciMLBase.DEStats()

    initial_int = init(prob, alg;
        tstops=sync_points,
        advance_to_tstop=true,
        init_args...
    )

    coarse_int = init(prob, alg;
        tstops=sync_points,
        advance_to_tstop=true,
        coarse_args...
    )

    ############################################################################
    ############################   START WORKERS    ############################
    ############################################################################
    for interval = 1:parareal_intervals
        # function call on worker
        fn = () -> solve_dist_worker(
            prob, alg,
            interval;
            maxit=maxit,
            info_channel=info_channels[interval],
            data_channel=data_channels[interval],
            t_0=sync_points[interval],
            t_end=sync_points[interval+1],
            fine_args...
        )

        if shared_memory
            worker_futures[interval] = @async fn()
        else
            worker_futures[interval] = @spawnat worker_map[interval] fn()
        end
    end

    ############################################################################
    ############################   INITIALIZATION    ###########################
    ############################################################################

    # handle first interval
    put!(data_channels[1], sync_values[1])
    put!(info_channels[1], SIGNAL_WORKER)
    reinit!(initial_int, sync_values[1], t0=sync_points[1])

    for interval = 1:parareal_intervals-1
        # initial integration
        step!(initial_int)
        coarse_result = initial_int.sol.u[end]

        # push result to worker
        wait_for_empty(info_channels[interval+1])
        put!(data_channels[interval+1], coarse_result)
        put!(info_channels[interval+1], SIGNAL_WORKER)

        sync_values[interval+1] = coarse_result
        coarse_prev[interval] = coarse_result
    end

    # add statistics from initialization
    _add_stats!(stats_total, initial_int.sol.stats)

    ############################################################################
    ##############################   MAIN LOOP    ##############################
    ############################################################################

    retcode = :Default
    iteration = 1
    maxit = min(maxit, parareal_intervals)

    while iteration <= maxit
        # if max. iterations reached, break here. Sequential update not needed
        if iteration >= maxit
            retcode = :MaxIters
            break
        end

        # first active interval is exact now.
        # no coarse integration, no update equation
        wait_for_signal(info_channels[iteration], SIGNAL_CONTROL, clear=true)
        fine_result = take!(data_channels[iteration])
        sync_errors[iteration] = 0  # parareal exactness
        sync_values[iteration+1] = fine_result
        # tell worker to stop
        wait_for_empty(info_channels[iteration])
        put!(info_channels[iteration], SIGNAL_DONE)

        for interval = iteration+1:parareal_intervals-1
            # coarse integration
            # reseting the integrator does not reset the statistics!
            _reset_stats!(coarse_int.sol.stats)
            reinit!(coarse_int, sync_values[interval], t0=sync_points[interval])
            step!(coarse_int)
            coarse_result = coarse_int.sol.u[end]

            # make sure fine result is available
            wait_for_signal(info_channels[interval], SIGNAL_CONTROL, clear=true)

            # get result
            fine_result = take!(data_channels[interval])

            # sync error
            sync_errors[interval] = norm(sync_values[interval+1], fine_result)

            # update equation
            sync_values[interval+1] = coarse_result + fine_result - coarse_prev[interval]

            # set coarse_prev for next iteration
            coarse_prev[interval] = coarse_result

            # add statistics
            _add_stats!(stats_total, coarse_int.sol.stats)
        end

        # take last result. Not required for update, but will mess up control flow otherwise
        wait_for_signal(info_channels[parareal_intervals], SIGNAL_CONTROL, clear=true)

        # get result
        take!(data_channels[parareal_intervals])

        # check for convergence
        max_err = maximum(sync_errors)

        if max_err < tol
            retcode = :MaxIters
            break
        end

        # update for next iteration
        for interval = iteration+1:parareal_intervals
            # make sure we can write
            wait_for_empty(info_channels[interval])

            # write data
            put!(data_channels[interval], sync_values[interval])
            put!(info_channels[interval], SIGNAL_WORKER)
        end

        iteration += 1
    end

    ############################################################################
    ##########################   COMBINE SOLUTIONS   ###########################
    ############################################################################

    # asynchronous task: merge full results
    #   depending on parareal settings, some to most workers finish early.
    #   transferring full solution and merging it to the overall solution
    #   can be done while the control node is waiting for other workers.
    merge = Threads.@spawn begin
        merged_sol = fetch(worker_futures[1])
        _add_stats!(stats_total, merged_sol.stats)

        for interval = 2:parareal_intervals
            sol = fetch(worker_futures[interval])
            _add_stats!(stats_total, sol.stats)
            merge_solution!(merged_sol, sol)
        end

        _reset_stats!(merged_sol.stats)
        _add_stats!(merged_sol.stats, stats_total)
        return merged_sol
    end

    # tell workers we are done
    force_signal.(info_channels[iteration+1:parareal_intervals], SIGNAL_DONE)

    # collect info
    info = (; retcode=retcode, iterations=iteration)

    return fetch(merge), info
end
