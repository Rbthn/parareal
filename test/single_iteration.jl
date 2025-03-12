# %% Setup
@everywhere function deriv!(du, u, p, t)
    du[:] = -1.0 .* u
    return nothing
end

t_0 = 0.0
t_end = 2.0
u_0 = [10.0]
dt = 1e-3
prob = ODEProblem(deriv!, u_0, (t_0, t_end), adaptive=false, saveat=dt, maxiters=typemax(Int))
alg = ImplicitEuler()



# reference: solve sequentially
sol_seq = OrdinaryDiffEq.solve(prob, alg; dt=dt)

#
# Make sure we get the correct result after a single iteration
# when coarse and fine propagator are chosen identical.
#

@testset "coarse = fine" begin

    parareal_args = (
        parareal_intervals=8,
        reltol=0.0,
        abstol=0.0,
        coarse_args=(; dt=dt),
        fine_args=(; dt=dt)
    )

    # %% Test 1: sync implementation
    @testset "sync. implementation" begin
        sol, stats = Parareal.solve_sync(
            prob, alg;
            dt=dt,
            parareal_args...
        )

        @test stats.iterations == 1
        @test stats.retcode == :Success
        @test sol.t == sol_seq.t
        @test sol.u ≈ sol_seq.u
    end


    # %% Test 2: async shared implementation
    @testset "async. threaded implementation" begin
        sol, stats = Parareal.solve_async(
            prob, alg;
            dt=dt,
            parareal_args...,
            shared_memory=true
        )

        @test stats.iterations == 1
        @test stats.retcode == :Success
        @test sol.t == sol_seq.t
        @test sol.u ≈ sol_seq.u
    end


    # %% Test 3: async dist implementation
    @testset "async. distributed implementation" begin
        sol, stats = Parareal.solve_async(
            prob, alg;
            dt=dt,
            parareal_args...,
            shared_memory=false
        )

        @test stats.iterations == 1
        @test stats.retcode == :Success
        @test sol.t == sol_seq.t
        @test sol.u ≈ sol_seq.u
    end
end


#
# Make sure we get the correct result after a single iteration
# when setting intervals=1
#

@testset "intervals = 1" begin
    parareal_args = (
        parareal_intervals=1,
        reltol=0.0,
        abstol=0.0,
        coarse_args=(; dt=2 * dt),
        fine_args=(; dt=dt)
    )

    # %% Test 1: sync implementation
    @testset "sync. implementation" begin
        sol, stats = Parareal.solve_sync(
            prob, alg;
            dt=dt,
            parareal_args...
        )

        @test stats.iterations == 1
        @test stats.retcode == :Success || stats.retcode == :MaxIters
        @test sol.t == sol_seq.t
        @test sol.u ≈ sol_seq.u
    end


    # %% Test 2: async shared implementation
    @testset "async. threaded implementation" begin
        sol, stats = Parareal.solve_async(
            prob, alg;
            dt=dt,
            parareal_args...,
            shared_memory=true
        )

        @test stats.iterations == 1
        @test stats.retcode == :Success || stats.retcode == :MaxIters
        @test sol.t == sol_seq.t
        @test sol.u ≈ sol_seq.u
    end


    # %% Test 3: async dist implementation
    @testset "async. distributed implementation" begin
        sol, stats = Parareal.solve_async(
            prob, alg;
            dt=dt,
            parareal_args...,
            shared_memory=false
        )

        @test stats.iterations == 1
        @test stats.retcode == :Success || stats.retcode == :MaxIters
        @test sol.t == sol_seq.t
        @test sol.u ≈ sol_seq.u
    end
end
