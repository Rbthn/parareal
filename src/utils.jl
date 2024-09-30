using DifferentialEquations

"""
    Merge contents of sol_2 into sol_1.
    This is a hack and not supported by DifferentialEquations!
"""
function merge_solution!(sol_1::ODESolution, sol_2::ODESolution)
    # sanity check: algorithm, dense, ... shold match
    if sol_1.alg != sol_2.alg
        throw(ArgumentError("Algorithm mismatch between partial solutions: expected $(sol_1.alg), got $(sol_2.alg)."))
    end
    if sol_1.dense != sol_2.dense
        throw(ArgumentError("Density mismatch between partial solutions: expected $(sol_1.dense), got $(sol_2.dense)."))
    end

    # solution
    append!(sol_1.t, sol_2.t[2:end])
    append!(sol_1.u, sol_2.u[2:end])

    return nothing
end

"""
    Add statistics entries of sol_2 to the ones of sol_1.
    This is a hack and not supported by DifferentialEquations!
"""
function sum_stats!(sol_1::ODESolution, sol_2::ODESolution)
    stat_fields = fieldnames(typeof(sol_1.stats))
    for stat in stat_fields
        stat1 = getfield(sol_1.stats, stat)
        stat2 = getfield(sol_2.stats, stat)
        setfield!(sol_1.stats, stat, stat1 + stat2)
    end

    return nothing
end