import SciMLBase
using Distributed

"""
    Merge contents of sol_2 into sol_1.
    This is a hack and not supported by DifferentialEquations!
"""
function merge_solution!(sol_1::SciMLBase.ODESolution, sol_2::SciMLBase.ODESolution)
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
    Add solver statistics from s2 to s1.
    This is a hack not directly supported by DifferentialEquations!
"""
function _add_stats!(s1::SciMLBase.DEStats, s2::SciMLBase.DEStats)
    for field in fieldnames(typeof(s2))
        value1 = getfield(s1, field)
        value2 = getfield(s2, field)
        setfield!(s1, field, value1 + value2)
    end
    return nothing
end

function _reset_stats!(stat::SciMLBase.DEStats)
    for field in fieldnames(typeof(stat))
        value = getfield(stat, field)
        setfield!(stat, field, zero(value))
    end
    return nothing
end

"""
    idle-wait on RemoteChannel `ch` until given signal `sig` is observed.
    Once observed, clear the signal from the channel iff `clear` is given, then return.
"""
function wait_for_signal(ch::Union{Channel,RemoteChannel}, sig; sleep_sec::Real=1e-3, clear=false)
    while true
        observed = fetch(ch)

        if observed == sig
            break
        else
            sleep(sleep_sec)
        end
    end

    if clear
        take!(ch)
    end

    return nothing
end

"""
    idle-wait until RemoteChannel `ch` is empty.
"""
function wait_for_empty(ch::Union{Channel,RemoteChannel}, fut=nothing; sleep_sec::Real=1e-3)
    while true
        empty = !isready(ch)

        if empty
            break
        else
            if !isnothing(fut)
                check_for_error(fut)
            end
            sleep(sleep_sec)
        end
    end

    return nothing
end

"""
    Write signal `sig` into RemoteChannel `ch`.
    If `ch` is empty, write right away.
    If not, clear one element, then write.
"""
function force_signal(ch::Union{Channel,RemoteChannel}, sig)
    if !isready(ch)
        put!(ch, sig)
    else
        take!(ch)
        put!(ch, sig)
    end
end

function check_for_error(fut::Future)
    if isready(fut)
        fetch(fut)
    end
    return nothing
end

function check_for_error(fut::Task)
    if istaskdone(fut)
        fetch(fut)
    end
    return nothing
end
