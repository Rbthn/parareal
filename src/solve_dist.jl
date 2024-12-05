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
    info_channel::RemoteChannel,
    data_channel::RemoteChannel,
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
            reinit!(int, initial_value, t0=t_0)
            step!(int)

            # write result to coarse channel. Used for update equation
            put!(data_channel, int.sol.u[end])
            put!(info_channel, SIGNAL_CONTROL)
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
"""
function solve_dist(
    prob::ODEProblem, alg;
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
    worker_map = Dict{Int,Int}()
    for i = 1:parareal_intervals
        worker_map[i] = workers()[i]
    end


    # set up info channel to each worker. Signals what to do with values in data_channel.
    info_channels = [RemoteChannel(
        () -> Channel{Int}(1), myid()
    ) for _ = 1:parareal_intervals]

    # set up channel for coarse result for each worker.
    data_channels = [RemoteChannel(
        () -> Channel{typeof(prob.u0)}(1), myid()
    ) for _ = 1:parareal_intervals]

    # vector of Futures for the function calls to workers.
    # Each worker will return its full solution via this Future
    worker_futures = Vector{Future}(undef, parareal_intervals)


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
        id = worker_map[interval]

        worker_futures[interval] = @spawnat id solve_dist_worker(
            prob, alg,
            interval;
            maxit=maxit,
            info_channel=info_channels[interval],
            data_channel=data_channels[interval],
            t_0=sync_points[interval],
            t_end=sync_points[interval+1],
            fine_args...
        )
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

    ############################################################################
    ##########################   SEQUENTIAL UPDATE    ##########################
    ############################################################################

    iteration = 0

    while iteration < maxit
        iteration += 1


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
        end

        # take last result. Not required for update, but will mess up control flow otherwise
        wait_for_signal(info_channels[parareal_intervals], SIGNAL_CONTROL, clear=true)

        # get result
        take!(data_channels[parareal_intervals])

        # check for convergence
        max_err = maximum(sync_errors)

        if max_err < tol
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

        # parareal exactness: signal that we're done
        sync_errors[1:iteration] .= 0
        for interval = 1:iteration
            wait_for_empty(info_channels[interval])
            put!(info_channels[interval], SIGNAL_DONE)
        end
    end

    force_signal.(info_channels, SIGNAL_DONE)

    ############################################################################
    ##########################   COMBINE SOLUTIONS   ###########################
    ############################################################################
    merged_sol = fetch(worker_futures[1])

    # TODO should we fetch all solutions first?
    for interval = 2:parareal_intervals
        sol = fetch(worker_futures[interval])
        merge_solution!(merged_sol, sol)
    end

    return merged_sol
end
