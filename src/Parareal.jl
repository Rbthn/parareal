module Parareal

include("utils.jl")

include("solve_sync.jl")
include("solve_async.jl")
include("solve.jl")
export solve

end
