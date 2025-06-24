using Test
using BenchmarkTools
using Expokit
using ExponentialUtilities
using StaticArrays


includet("../modules/OpticalCavity.jl")
includet("../modules/Thermodynamics.jl")
includet("../modules/Phaseonium.jl")
includet("../modules/BosonicOperators.jl")
includet("../modules/Measurements.jl")

includet("../modules/MasterEquations.jl")
includet("../src/main.jl")

dir = "tests";
config, ndims = init(dir)
println("Dims: $ndims")

state = load_or_create(dir, config)
rho = state.ρ
c1 = state.c₁ 
c2 = state.c₂
c1_fast = deepcopy(c1)
c2_fast = deepcopy(c2)

dt = config["dt"]

process = "Expansion"
stop1 = stop2 = false

forces = [c1.expanding_force, c2.expanding_force]

function test_benchmark()
    dir = "tests"
    config, dims = init(dir)
    println("Dims: $dims")

    state = load_or_create(dir, config)
    rho = state.ρ
    c1 = state.c₁ 
    c2 = state.c₂
    c1_fast = deepcopy(c1)
    c2_fast = deepcopy(c2)

    dt = config["dt"]

    process = "Expansion"
    stop1 = stop2 = false

    forces = [c1.expanding_force, c2.expanding_force]

    cache = make_cache(dims)
    cache_fast = make_cache_fast(dims, forces)
    
    args = (cache, dt, 0, process, stop1, stop2)
    args_fast = (cache_fast, dt, 0, process, stop1, stop2)
    rho, c1, c2 = adiabatic_loop(rho, c1, c2, args, verbose=true)
    rho_fast, c1_fast, c2_fast = adiabatic_loop_fast(rho, c1_fast, c2_fast, args_fast, verbose=true)

    @testset "State" begin
        @test norm(rho - rho_fast) < 1e-10
        @test tr(rho) ≈ 1.0
        @test tr(rho_fast) ≈ 1.0
    end
    @testset "Cavities" begin
        @test c1.length ≈ c1_fast.length
        @test c2.length ≈ c2_fast.length
        @test c1.acceleration ≈ c1_fast.acceleration
        @test c2.acceleration ≈ c2_fast.acceleration
    end

    # b1 = @benchmarkable adiabaticevolve($rho, ($c1, $c2), $args...)
    b1 = @benchmarkable adiabatic_loop($rho, $c1, $c2, $args)
    # b2 = @benchmarkable adiabaticevolve_fast($rho, ($c1_fast, $c2_fast), $args_fast...)
    b2 = @benchmarkable adiabatic_loop_fast($rho, $c1_fast, $c2_fast, $args_fast)
    tune!(b1)
    tune!(b2)

    display(run(b1, seconds=60))
    display(run(b2, seconds=60))

end


function make_cache(dims)
    # Operators are defined on one subspace
    identity_matrix = spdiagm(ones(dims))
    opa = BosonicOperators.destroy(dims, true)
    opad = BosonicOperators.create(dims, true) 
    # Pressure Operator
    # Decomposed in three parts (constant and rotating)
    n = opad * opa
    π_a = opa * opa
    π_ad = opad * opad

    # Preallocate variables with reduce precision
    U = spzeros(ComplexF64, dims^2, dims^2)

    return (U, identity_matrix, n, π_a, π_ad)
end


function make_cache_fast(dims, forces)
    # Operators are defined on one subspace
    opa = BosonicOperators.destroy(dims, true)
    opad = BosonicOperators.create(dims, true)
    n = opad * opa
    identity_matrix = spdiagm(ones(dims))

    return (;
        U = spzeros(ComplexF64, dims^2, dims^2),
        Ud = spzeros(ComplexF64, dims^2, dims^2),
        temp = zeros(ComplexF64, dims^2, dims^2),
        idd = identity_matrix,
        n = n,
        π_a = opa * opa,
        π_ad = opad * opad,
        π_op = Matrix{ComplexF64}(undef, dims, dims),
        _h = sparse(n .+ 0.5 .* identity_matrix),
        h1 = spzeros(ComplexF64, dims, dims),
        h2 = spzeros(ComplexF64, dims, dims),
        h = spzeros(ComplexF64, dims^2, dims^2),
        a1 = 0,
        a2 = 0,
        p1 = 0,
        p2 = 0,
        force1 = forces[1],
        force2 = forces[2],
    )
end


function adiabatic_loop(ρ, c1, c2, args; verbose=false)
    verbose && print("Starting with lengths: $(c1.length) - $(c2.length)\n")
    c_1, c_2 = deepcopy(c1), deepcopy(c2)
    for i in range(1, 100)
        ρ, c_1, c_2 = adiabaticevolve(ρ, (c_1, c_2), args...)
        verbose && print(" Final lengths 1: $(c_1.length) - $(c_2.length)\n")
    end
    return ρ, c_1, c_2
end


function adiabatic_loop_fast(ρ, c1, c2, args; verbose=false)
    verbose && print("Starting with lengths: $(c1.length) - $(c2.length)\n")
    c_1, c_2 = deepcopy(c1), deepcopy(c2)
    for i in range(1, 100)
        ρ, c_1, c_2 = adiabaticevolve_fast(ρ, (c_1, c_2), args...)
        verbose && print(" Final lengths 2: $(c_1.length) - $(c_2.length)\n")
    end
    return ρ, c_1, c_2
end


function adiabaticevolve(ρ, cavities, cache, Δt, t, process, stop1, stop2)
    U, idd, n, π_a, π_ad = cache
    
    Δt² = Δt^2  

    c1, c2 = cavities
    α0 = c1.α

    a1 = c1.acceleration
    a2 = c2.acceleration
    
    # Move the cavity walls
    if !stop1
        c1.length += 0.5 * c1.acceleration * Δt²
        # Constrain the cavity movement blocking the walls
        c1.length = clamp(c1.length, c1.l_min, c1.l_max)
    end
    if !stop2
        c2.length += 0.5 * c2.acceleration * Δt²
        # Constrain the cavity movement blocking the walls
        c2.length = clamp(c2.length, c2.l_min, c2.l_max)
    end
    
    
    # Update energies
    ω₁ = α0 / c1.length
    ω₂ = α0 / c2.length
    h1 = ω₁ .* (n .+ 0.5 * idd)
    h2 = ω₂ .* (n .+ 0.5 * idd)
    h = kron(h1, h2)
    
    # Evolve the System
    U .= Expokit.padm(-im .* h)
    ρ = U * ρ * U'
    
    # Update pressure and acceleration
    π₁ = (2 * n + idd)  - (π_a * exp(-2*im*ω₁*t)) - (π_ad * exp(2*im*ω₁*t))
    π₂ = (2 * n + idd)  - (π_a * exp(-2*im*ω₂*t)) - (π_ad * exp(2*im*ω₂*t))
    p1 = Measurements.pressure(ρ, π₁, idd, α0, c1.length, c1.surface; s=1)
    p2 = Measurements.pressure(ρ, π₂, idd, α0, c2.length, c2.surface; s=2)
    
    if process == "Expansion"
        force1 = c1.expanding_force
        force2 = c2.expanding_force
    else
        force1 = c1.compressing_force
        force2 = c2.compressing_force
    end

    a1 = (p1 * c1.surface - force1) / c1.mass
    a2 = (p2 * c2.surface - force2) / c2.mass
    
    # println("p1:$p1 - p2:$p2\nf1:$force1/$a1 - f2:$force2/$a2")
    
    if norm(a1) <= 0.01 || norm(a2) <= 0.01
        error(
            "One cavity is almost still during $process \
            with force $force1/$force2 and pressure $p1/$p2")
    end
    if process == "Expansion" && (a1 < 0 || a2 < 0)
        error("One cavity is going backward during expansion! $a1/$a2-$p1/$p2")
    elseif process == "Contraction" && (a1 > 0 || a2 > 0)
        error("One cavity is going forward during contraction!")
    end
    if c1.acceleration * a1 < 0 || c2.acceleration * a2 < 0
        error("One cavity changed direction during $process !")
    end
    c1.acceleration = a1
    c2.acceleration = a2

    return ρ, c1, c2
end

function adiabaticevolve_fast(ρ, cavities, cache, Δt, t, process, stop1, stop2)
    U = cache.U
    Ud = cache.Ud
    idd = cache.idd
    _h = cache._h
    h1 = cache.h1
    h2 = cache.h2
    h = cache.h
    n = cache.n
    π_a = cache.π_a
    π_ad = cache.π_ad
    π_op = cache.π_op
    a1 = cache.a1
    a2 = cache.a2
    p1 = cache.p1
    p2 = cache.p2
    force1 = cache.force1
    force2 = cache.force2
    temp = cache.temp
    
    Δt² = Δt^2  

    c1, c2 = cavities
    α0 = c1.α
    
    # Move the cavity walls
    if !stop1
        c1.length += 0.5 * c1.acceleration * Δt²
        # Constrain the cavity movement blocking the walls
        c1.length = clamp(c1.length, c1.l_min, c1.l_max)
    end
    if !stop2
        c2.length += 0.5 * c2.acceleration * Δt²
        # Constrain the cavity movement blocking the walls
        c2.length = clamp(c2.length, c2.l_min, c2.l_max)
    end
    
    
    # Update energies
    ω₁ = α0 / c1.length
    ω₂ = α0 / c2.length
    @. h1 = ω₁ * _h
    @. h2 = ω₂ * _h
    kron!(h, h1, h2)
    
    # Evolve the System
    # U .= padm(-im .* h)
    U .= Expokit.padm(-im * h)  # Use in-place matrix exponential
    Ud .= U'
    mul!(temp, U, ρ) # ρ_temp = U * ρ
    mul!(ρ, temp, Ud) # ρ = ρ_temp * Ud
    
    # Update pressure and acceleration
    @. π_op = (2 * n + idd)  - (π_a * exp(-2*im*ω₁*t)) - (π_ad * exp(2*im*ω₁*t))
    p1 = Measurements.pressure(ρ, π_op, idd, α0, c1.length, c1.surface; s=1)
    @. π_op = (2 * n + idd)  - (π_a * exp(-2*im*ω₂*t)) - (π_ad * exp(2*im*ω₂*t))
    p2 = Measurements.pressure(ρ, π_op, idd, α0, c2.length, c2.surface; s=2)

    a1 = (p1 * c1.surface - force1) / c1.mass
    a2 = (p2 * c2.surface - force2) / c2.mass
    
    # println("p1:$p1 - p2:$p2\nf1:$force1/$a1 - f2:$force2/$a2")
    
    # if norm(a1) <= 0.01 || norm(a2) <= 0.01
    #     error(
    #         "One cavity is almost still during $process \
    #         with force $force1/$force2 and pressure $p1/$p2")
    # end
    # if process == "Expansion" && (a1 < 0 || a2 < 0)
    #     error("One cavity is going backward during expansion!")
    # elseif process == "Contraction" && (a1 > 0 || a2 > 0)
    #     error("One cavity is going forward during contraction!")
    # end
    # if c1.acceleration * a1 < 0 || c2.acceleration * a2 < 0
    #     error("One cavity changed direction during $process!")
    # end
    c1.acceleration = a1
    c2.acceleration = a2

    return ρ, c1, c2
end
