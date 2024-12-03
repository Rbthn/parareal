import SciMLBase
using DifferentialEquations
using Distributed

"""
    Helper function called on each worker.
    Receives updated initial values from control node,
    performs integration and sends end value back to control node.
"""
function solve_dist_worker(
    prob::ODEProblem,
    alg,
    interval::Int;
    parareal_intervals::Int,
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
    while iteration < maxit
        # wait for signal from control node
        signal = fetch(info_channel)

        # 0: control node's turn to perform coarse update
        # 1: our turn to perform fine integration
        # -1: this node is done
        if signal == -1
            break
        elseif signal == 1
            take!(info_channel)
            iteration += 1

            initial_value = take!(data_channel)

            # fine integration
            reinit!(int, initial_value, t0=t_0)
            step!(int)

            # write result to coarse channel. Used for update equation
            put!(data_channel, int.sol.u[end])
            put!(info_channel, 0)
        elseif signal == 0
            sleep(1)
            continue
        else
            sleep(1)
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
    # only holds one integer:
    #   1: worker's turn to read result, do computation, write result
    #   0: control node's turn to read  result, do computation, write result
    #   -1: algorithm terminated, worker should return
    info_channels = [RemoteChannel(
        () -> Channel{Int}(1), myid()
    ) for _ = 1:parareal_intervals]

    # set up channel for coarse result for each worker.
    data_channels = [RemoteChannel(
        () -> Channel{typeof(prob.u0)}(1), myid()
    ) for _ = 1:parareal_intervals]


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

        # coarse channel
        data_ch = data_channels[interval]

        # info channel
        info_ch = info_channels[interval]

        @spawnat id solve_dist_worker(
            prob, alg,
            interval;
            parareal_intervals=parareal_intervals,
            maxit=maxit,
            info_channel=info_ch,
            data_channel=data_ch,
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
    put!(info_channels[1], 1)
    reinit!(initial_int, sync_values[1], t0=sync_points[1])

    for interval = 1:parareal_intervals-1
        # initial integration
        step!(initial_int)
        coarse_result = initial_int.sol.u[end]

        # push result to worker
        wait_for_empty(info_channels[interval+1])
        put!(data_channels[interval+1], coarse_result)
        put!(info_channels[interval+1], 1)

        sync_values[interval+1] = coarse_result
        coarse_prev[interval] = coarse_result
    end

    # we wait for fine integrators to finish
    for interval = 1:parareal_intervals
        wait_for_signal(info_channels[interval], 0)
    end


    ############################################################################
    ##########################   SEQUENTIAL UPDATE    ##########################
    ############################################################################

    iteration = 0

    while iteration < maxit
        iteration += 1


        # can we skip coarse integration for interval iteration?
        for interval = iteration:parareal_intervals-1
            # coarse integration
            reinit!(coarse_int, sync_values[interval], t0=sync_points[interval])
            step!(coarse_int)
            coarse_result = coarse_int.sol.u[end]

            # make sure fine result is available
            wait_for_signal(info_channels[interval], 0, clear=true)

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
        wait_for_signal(info_channels[parareal_intervals], 0, clear=true)

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
            put!(info_channels[interval], 1)
        end

        # parareal exactness: signal that we're done
        sync_errors[1:iteration] .= 0
        for interval = 1:iteration
            wait_for_empty(info_channels[interval])
            put!(info_channels[interval], -1)
        end
    end


    ## TODO stitch solutions

    return nothing
end
