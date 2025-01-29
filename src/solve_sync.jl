import SciMLBase
using OrdinaryDiffEq

"""
    Solve ODEProblem using the Parareal algorithm.
    Use thread-parallel for-loop for parallel integration.
"""
function solve_sync(prob::SciMLBase.ODEProblem, alg;
    parareal_intervals::Int,
    reltol=1e-3::Float64,
    abstol=1e-6::Float64,
    norm=(x, y) -> maximum(abs.(x - y)),
    maxit=parareal_intervals::Int,
    coarse_args=(),
    fine_args=(),
    init_args=coarse_args,
    kwargs...
)

    ############################################################################
    ################################   SETUP   #################################
    ############################################################################

    # divide time span into multiple intervals
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
    sync_errors_abs = fill(Inf, parareal_intervals - 1)
    sync_errors_rel = fill(Inf, parareal_intervals - 1)

    # statistics
    stats_total = SciMLBase.DEStats()
    nsolve_seq = 0

    if parareal_intervals > 1
        # initial coarse solve. Make sure to step to the sync points.
        initial_int = init(prob, alg;
            tstops=sync_points,
            advance_to_tstop=true,
            kwargs...,
            init_args...,
        )

        for i = 1:parareal_intervals-1
            step!(initial_int)
            sync_values[i+1] = copy(initial_int.sol.u[end])
            coarse_prev[i] = copy(initial_int.sol.u[end])
        end

        # add statistics from initialization
        _add_stats!(stats_total, initial_int.sol.stats)
        nsolve_seq += initial_int.stats.nsolve


        # coarse integrator
        coarse_int = init(prob, alg;
            tstops=sync_points,         # make sure to step to sync points exactly
            advance_to_tstop=true,      # this makes step!() step until reaching next sync point
            kwargs...,
            coarse_args...,
        )
    end

    # fine integrators
    fine_ints = [init(prob, alg;
        tstops=sync_points[i:i+1],  # make sure to step to sync points exactly
        advance_to_tstop=true,      # this makes step!() step until reaching
        kwargs...,
        fine_args...,
    ) for i = 1:parareal_intervals]


    ############################################################################
    ##########################   PARAREAL ITERATION   ##########################
    ############################################################################

    retcode = :Default
    iteration = 1
    maxit = min(maxit, parareal_intervals)

    while iteration <= maxit
        # thread-parallel loop
        Threads.@threads for i = iteration:parareal_intervals
            fine_int = fine_ints[i]
            _reset_stats!(fine_int.sol.stats)
            reinit!(fine_int, sync_values[i], t0=sync_points[i])
            step!(fine_int)

            fine_result = fine_int.sol.u[end]
            if i != parareal_intervals
                sync_errors_abs[i] = norm(sync_values[i+1], fine_result)
                sync_errors_rel[i] = sync_errors_abs[i] / norm(zeros(size(fine_result)), fine_result)
            end
        end

        # add statistics
        nsolve_fine_max = 0
        for i = iteration:parareal_intervals
            stat = fine_ints[i].sol.stats
            _add_stats!(stats_total, stat)

            nsolve_fine_max = max(nsolve_fine_max, stat.nsolve)
        end
        nsolve_seq += nsolve_fine_max

        # check for convergence
        if !isempty(sync_errors_abs) && (maximum(sync_errors_abs) < abstol || maximum(sync_errors_rel) < reltol)
            retcode = :Success
            break
        end

        # if max. iterations reached, break here. Sequential update not needed
        if iteration >= maxit
            retcode = :MaxIters
            break
        end

        # first active interval is exact now.
        # no coarse integration, no update equation
        sync_errors_abs[iteration] = sync_errors_rel[iteration] = 0
        sync_values[iteration+1] = fine_ints[iteration].sol.u[end]

        # sequential coarse solve
        for i = iteration+1:parareal_intervals-1
            _reset_stats!(coarse_int.sol.stats)
            reinit!(coarse_int, sync_values[i], t0=sync_points[i])
            step!(coarse_int)
            coarse_res = coarse_int.sol.u[end]

            # update equation
            sync_values[i+1] = coarse_res + fine_ints[i].u - coarse_prev[i]
            coarse_prev[i] = coarse_res

            # add statistics
            _add_stats!(stats_total, coarse_int.sol.stats)
            nsolve_seq += coarse_int.stats.nsolve
        end

        iteration += 1
    end


    ############################################################################
    ##########################   COMBINE SOLUTIONS   ###########################
    ############################################################################

    sols = [int.sol for int in fine_ints]
    merged_sol = sols[1]

    # combine fine solutions
    for sol in sols[2:end]
        merge_solution!(merged_sol, sol)
    end

    # set statistics
    _reset_stats!(merged_sol.stats)
    _add_stats!(merged_sol.stats, stats_total)

    # collect info
    info = (; nsolve_seq=nsolve_seq, retcode=retcode, iterations=iteration, abs_error=maximum(sync_errors_abs), rel_error=maximum(sync_errors_rel))

    return merged_sol, info
end
