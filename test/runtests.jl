using Distributed, OrdinaryDiffEq
using Parareal
using Test

@testset verbose = true "Parareal" begin
    include("input_validation.jl")
    include("single_iteration.jl")
    include("full_iteration.jl")
end
