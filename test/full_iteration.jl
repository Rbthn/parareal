#
# Make sure that we obtain the same result if we iterate all the way through.
#

@testset "Full iteration" begin
    # %% Setup
    function deriv!(du, u, p, t)
        du[:] = -1.0 .* u
        return nothing
    end

    t_0 = 0.0
    t_end = 2.0
    u_0 = [10.0]
    dt_fine = 1e-5
    dt_coarse = 1e-3
    prob = ODEProblem(deriv!, u_0, (t_0, t_end), adaptive=false, saveat=dt_fine, maxiters=typemax(Int))
    alg = ImplicitEuler()

    parareal_args = (
        parareal_intervals=8,
        reltol=0.0,
        abstol=0.0,
        coarse_args=(; dt=dt_coarse)
    )

    # reference: solve sequentially
    sol_seq = OrdinaryDiffEq.solve(prob, alg; dt=dt_fine)


    # %% Test 1: sync implementation
    @testset "Sync. implementation" begin
        sol, _ = Parareal.solve_sync(
            prob, alg;
            dt=dt_fine,
            parareal_args...
        )

        @test sol.t == sol_seq.t
        @test sol.u ≈ sol_seq.u
    end


    # %% Test 2: async shared implementation
    @testset "Async. threaded implementation" begin
        sol, _ = Parareal.solve_async(
            prob, alg;
            dt=dt_fine,
            parareal_args...,
            shared_memory=true
        )

        @test sol.t == sol_seq.t
        @test sol.u ≈ sol_seq.u
    end


    # %% Test 3: async dist implementation
    @testset "Async. distributed implementation" begin
        sol, _ = Parareal.solve_async(
            prob, alg;
            dt=dt_fine,
            parareal_args...,
            shared_memory=false
        )

        @test sol.t == sol_seq.t
        @test sol.u ≈ sol_seq.u
    end
end
