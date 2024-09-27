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
    fine_args=())

    ############################################################################
    ################################   SETUP   #################################
    ############################################################################

    # divide time span into multiple intervals
    sync_points = LinRange(
        prob.tspan[1], prob.tspan[end], parareal_intervals + 1)

    # solution values at sync points. These are modified in the
    # parareal update eqn. and used as initial values in subsequent iterations.
    sync_values = Vector{typeof(prob.u0)}(undef, parareal_intervals + 1)

    # results of coarse propagator at sync points from prev. iteration.
    # saved to avoid re-computing
    coarse_prev = Vector{typeof(prob.u0)}(undef, parareal_intervals + 1)

    # errors at sync points. Difference between left and right solution,
    # measured by given norm
    sync_errors = fill(Inf, parareal_intervals)

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

    # initial coarse solve. Make sure to step to the sync points.
    sync_values[1] = prob.u0
    for i = 1:parareal_intervals
        step!(coarse_int)
        sync_values[i+1] = coarse_prev[i] = coarse_int.sol.u[end]
    end

    ############################################################################
    ##########################   PARAREAL ITERATION   ##########################
    ############################################################################


    ############################################################################
    ##########################   COMBINE SOLUTIONS   ###########################
    ############################################################################


    return nothing
end
