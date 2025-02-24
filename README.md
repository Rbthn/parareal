# Parareal

Implementation of the [Parareal](https://en.wikipedia.org/wiki/Parareal) algorithm in Julia.

## Prerequisites

- [OrdinaryDiffEq](https://github.com/SciML/OrdinaryDiffEq.jl)
- [Distributed](https://github.com/JuliaLang/Distributed.jl)

## Installation

```julia
pkg> activate .
pkg> add <URL>
```

## Getting started

This implementation supports two parallelization mechanisms:

### Worker-based model using [Distributed](https://docs.julialang.org/en/v1/stdlib/Distributed/)

- Default choice
- Add worker processes (via `addprocs(N)`)
- WARNING: if your problem makes use of global variables (common when loading a script into global scope), they need to be made available to all workers. This can be done by prefixing their declaration with `@everywhere`
- This method has higher communication overhead but suffers less from frequent garbage collection. It is expected to outperform the threaded implementation for allocation-heavy problems (e.g. many timesteps).
- Workers may be distributed among different machines (e.g. using [ClusterManagers](https://github.com/JuliaParallel/ClusterManagers.jl).

### Thread-based model using [Threads](https://docs.julialang.org/en/v1/manual/multi-threading/)

- activated with keyword-argument `shared_memory=true`
- start Julia with multiple threads: `julia -t N`
- This method has lower communication overhead, but its performance suffers from frequent garbage collection. It is thus expected to be slower than the distributed implemetation for most problems.
