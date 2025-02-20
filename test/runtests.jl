using Distributed, OrdinaryDiffEq
using Parareal
using Test

@testset "tests" begin
    include("input_validation.jl")
    include("single_iteration.jl")
    include("full_iteration.jl")
end
