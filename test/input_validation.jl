#
# Make sure that faulty inputs are handled correctly.
# In particular, waiting for a future may hang forever
# if the underlying worker code throws an error
#

using Distributed

# consider test failed if we don't get a result after wait_timeout
const wait_timeout = 15.0

@testset "missing parameters" begin
    try
        # %% Setup
        procs = addprocs(4)
        @everywhere using Parareal

        # using parameter deliberately not defined on worker
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


        # if not handled correctly, worker tasks will fail and main task
        # will be stuck waiting for results -> run async
        fut = @async Parareal.solve_async(
            prob, alg;
            parareal_args...,
            shared_memory=false
        )

        # wait for result or timeout
        sleep(wait_timeout)

        @test istaskdone(fut)
        @test_throws Exception fetch(fut)
    finally
        rmprocs(procs)
    end
end
