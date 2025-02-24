# %% code loading
using Parareal


# %% add workers
using Distributed
const N = 8
procs = addprocs(N)


# %% load dependencies
@everywhere begin
    using Parareal
    using OrdinaryDiffEq
end


# %% define problem
@everywhere begin
    decay_rate = -2.0

    # use in-place version to reduce allocations
    function deriv!(du, u, p, t)
        du[:] = decay_rate .* u
        return nothing
    end
end

t_0 = 0.0
t_end = 8.0
dt = 1e-2
u_0 = [10.0]

prob = ODEProblem(deriv!, u_0, (t_0, t_end), adaptive=false, saveat=dt)


# %% define parareal options
parareal_args = (;
    parareal_intervals=N,
    reltol=1e-3,
    abstol=1e-3,
    coarse_args=(; dt=10 * dt),
    fine_args=(; dt=dt),
    shared_memory=false,
)


# %% solve
sol, stats = Parareal.solve(prob, ImplicitEuler(); parareal_args...)


# %% cleanup
rmprocs(procs)
