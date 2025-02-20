module Parareal

include("utils.jl")

include("solve_sync.jl")
include("solve_async.jl")

solve = solve_async
export solve

end
