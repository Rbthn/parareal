import SciMLBase
using OrdinaryDiffEq
using Logging

"""
    solve_sync(prob, alg; <keyword arguments>)

Solve given ODEProblem `prob` with Parareal, using the integration scheme `alg`.
Parallel ("fine") integration is delegated to other threads via a thread-parallel `for`-loop, sequential ("coarse") integration is performed
on the thread this function is called on.

WARNING: This implementation ignores some optimizations in favor
of an easier to understand, non-asynchronous program flow.
It should only be used as a reference, as [`solve_async`](@ref) is expected
to be faster.

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

See also: [`solve_async`](@ref).
"""
function solve_sync(
    prob::SciMLBase.ODEProblem, alg;
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
    nsolve_seq = 0
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

        # initial integration
        @debug "Starting sequential initialization"
        for interval = 1:parareal_intervals-1
            step!(initial_int)
            coarse_result = initial_int.sol.u[end]

            sync_values[interval+1] = copy(coarse_result)
            coarse_prev[interval] = copy(coarse_result)
        end

        # add statistics from initialization
        if statistics
            _add_stats!(stats_total, initial_int.sol.stats)
            nsolve_seq += initial_int.stats.nsolve
        end
    end

    # fine integrators
    fine_ints = [init(prob, alg;
        kwargs...,
        fine_args...,
        tstops=sync_points[i:i+1],  # make sure to step to sync points exactly
        advance_to_tstop=true,      # this makes step!() step until reaching
    ) for i = 1:parareal_intervals]


    ############################################################################
    ##########################   PARAREAL ITERATION   ##########################
    ############################################################################

    retcode = :Default
    iteration = 1
    maxit = min(maxit, parareal_intervals)

    while iteration <= maxit
        @debug "Iteration $iteration in main Parareal loop"

        # thread-parallel loop
        Threads.@threads for i = iteration:parareal_intervals
            fine_int = fine_ints[i]
            statistics && _reset_stats!(fine_int.sol.stats)
            reinit!(fine_int, sync_values[i], t0=sync_points[i])
            step!(fine_int)

            fine_result = fine_int.sol.u[end]
            if i != parareal_intervals
                sync_errors_abs[i] = norm(sync_values[i+1], fine_result)
                sync_errors_rel[i] = sync_errors_abs[i] / norm(zeros(size(fine_result)), fine_result)
            end
        end

        # add statistics
        if statistics
            nsolve_fine_max = 0
            for i = iteration:parareal_intervals
                stat = fine_ints[i].sol.stats
                _add_stats!(stats_total, stat)

                nsolve_fine_max = max(nsolve_fine_max, stat.nsolve)
            end
            nsolve_seq += nsolve_fine_max
        end

        # check for convergence
        if !isempty(sync_errors_abs) && (maximum(sync_errors_abs) <= abstol || maximum(sync_errors_rel) <= reltol)
            @debug "Parareal synchronization errors below tolerance"
            retcode = :Success
            break
        end

        # if max. iterations reached, break here. Sequential update not needed
        if iteration >= maxit
            @debug "Max. iterations reached"
            retcode = :MaxIters
            break
        end

        # first active interval is exact now.
        # no coarse integration, no update equation
        sync_errors_abs[iteration] = sync_errors_rel[iteration] = 0
        sync_values[iteration+1] = fine_ints[iteration].sol.u[end]

        # sequential coarse solve
        @debug "Starting sequential update in iteration $iteration"
        for i = iteration+1:parareal_intervals-1
            statistics && _reset_stats!(coarse_int.sol.stats)
            reinit!(coarse_int, sync_values[i], t0=sync_points[i])
            step!(coarse_int)
            coarse_res = coarse_int.sol.u[end]

            # update equation
            sync_values[i+1] = copy(coarse_res) + fine_ints[i].u - coarse_prev[i]
            coarse_prev[i] = copy(coarse_res)

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
    statistics && _add_stats!(merged_sol.stats, stats_total)

    # collect info
    info = (;
        retcode=retcode,
        iterations=iteration,
        abs_error=maximum(sync_errors_abs, init=0.0),
        rel_error=maximum(sync_errors_rel, init=0.0),
        nsolve_seq=nsolve_seq,
    )

    return merged_sol, info
end
