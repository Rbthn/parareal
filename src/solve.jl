using DifferentialEquations

"""
    Solve ODEProblem using the Parareal algorithm.
"""
function solve(prob::ODEProblem, alg;
    parareal_intervals::Int,
    tol=1e-3::Float64,
    norm=(x, y) -> maximum(abs.(x - y)),
    maxit=parareal_intervals::Int,
    coarse_args=(),
    fine_args=(),
    init_args=coarse_args)

    ############################################################################
    ################################   SETUP   #################################
    ############################################################################

    # divide time span into multiple intervals
    sync_points = LinRange(
        prob.tspan[1], prob.tspan[end], parareal_intervals + 1)

    # solution values at sync points. These are modified in the
    # parareal update eqn. and used as initial values in subsequent iterations.
    sync_values = Vector{typeof(prob.u0)}(undef, parareal_intervals)

    # results of coarse propagator at sync points from prev. iteration.
    # saved to avoid re-computing.
    # coarse_prev[i] contains value at sync_points[i+1]
    # obtained from coarse propagation started at sync_points[i]
    coarse_prev = Vector{typeof(prob.u0)}(undef, parareal_intervals - 1)

    # errors at sync points. Difference between left and right solution,
    # measured by given norm
    sync_errors = fill(Inf, parareal_intervals - 1)


    # initial coarse solve. Make sure to step to the sync points.
    initial_int = init(prob, alg;
        tstops=sync_points,
        advance_to_tstop=true,
        init_args...)

    sync_values[1] = prob.u0
    for i = 1:parareal_intervals-1
        step!(initial_int)
        sync_values[i+1] = copy(initial_int.sol.u[end])
        coarse_prev[i] = copy(initial_int.sol.u[end])
    end

    # coarse integrator
    coarse_int = init(prob, alg;
        tstops=sync_points,         # make sure to step to sync points exactly
        advance_to_tstop=true,      # this makes step!() step until reaching next sync point
        coarse_args...)

    # fine integrators
    fine_ints = [init(prob, alg;
        tstops=sync_points[i:i+1],  # make sure to step to sync points exactly
        advance_to_tstop=true,      # this makes step!() step until reaching
        fine_args...) for i = 1:parareal_intervals]

    ############################################################################
    ##########################   PARAREAL ITERATION   ##########################
    ############################################################################


    for iteration = 1:maxit
        # thread-parallel loop
        Threads.@threads for i = iteration:parareal_intervals
            fine_int = fine_ints[i]
            reinit!(fine_int, sync_values[i], t0=sync_points[i])
            step!(fine_int)

            if i != parareal_intervals
                sync_errors[i] = norm(sync_values[i+1], fine_int.sol.u[end])
            end
        end

        # parareal exactness
        sync_errors[1:iteration-1] .= 0

        # check for convergence
        if maximum(sync_errors) <= tol
            break
        end

        # sequential coarse solve
        for i = iteration:parareal_intervals-1
            reinit!(coarse_int, sync_values[i], t0=sync_points[i])
            step!(coarse_int)

            # update equation
            sync_values[i+1] = coarse_int.u + fine_ints[i].u - coarse_prev[i]
        end
    end


    ############################################################################
    ##########################   COMBINE SOLUTIONS   ###########################
    ############################################################################

    sols = [int.sol for int in fine_ints]
    merged_sol = sols[1]

    # combine fine solutions
    for sol_2 in sols[2:end]
        merge_solution!(merged_sol, sol_2)
        sum_stats!(merged_sol, sol_2)
    end

    # add statistics from coarse solver
    sum_stats!(merged_sol, coarse_int.sol)

    return merged_sol
end