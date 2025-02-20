#
# Make sure that faulty inputs are handled correctly.
# In particular, waiting for a future may hang forever
# if the underlying worker code throws an error
#

using Distributed

# consider test failed if we don't get a result after 5 seconds
const wait_timeout = 15.0
const parameter = -1.0

@testset "global parameter not defined @everywhere" begin
    global parameter
    try
        # %% Setup
        procs = addprocs(4)
        @everywhere using Parareal

        # using parameter deliberately defined on control process only
        function deriv!(du, u, p, t)
            du[:] = parameter .* u
            return nothing
        end

        t_0 = 0.0
        t_end = 1.0
        u_0 = [10.0]
        dt_coarse = 0.25
        dt_fine = 0.125

        prob = ODEProblem(deriv!, u_0, (t_0, t_end), adaptive=false, saveat=dt_fine, maxiters=typemax(Int))
        alg = ImplicitEuler()

        parareal_args = (
            parareal_intervals=4,
            reltol=0.0,
            abstol=0.0,
            coarse_args=(; dt=dt_coarse),
            fine_args=(; dt=dt_fine)
        )

        fut = @async Parareal.solve_async(
            prob, alg;
            parareal_args...,
            shared_memory=false
        )
        sleep(wait_timeout)

        @test istaskdone(fut)
        @test_throws Exception fetch(fut)
    finally
        rmprocs(procs)
    end
end
