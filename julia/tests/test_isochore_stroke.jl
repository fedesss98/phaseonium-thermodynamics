using Test
using BenchmarkTools
using Expokit
using StaticArrays


includet("../modules/OpticalCavity.jl")
includet("../modules/Thermodynamics.jl")
includet("../modules/Phaseonium.jl")
includet("../modules/BosonicOperators.jl")
includet("../modules/Measurements.jl")

includet("../modules/MasterEquations.jl")
includet("../src/main.jl")


function test_benchmark()
    dir = "tests";
    config, ndims = init(dir)
    println(ndims)

    state = load_or_create(dir, config)
    dt = config["dt"]
    
    process = "Expansion"
    stop1 = stop2 = false

    forces = [state.c₁.expanding_force, state.c₂.expanding_force]

    cache = make_cache(ndims)
    cache_fast = make_cache_fast(ndims, forces)
    
    args = (cache, dt, 0, process, stop1, stop2)
    ρ, c1, c2 = evolution_loop(state.ρ, state.c₁, state.c₂, args)
    args_fast = (cache_fast, dt, 0, process, stop1, stop2)
    ρ_fast, c1_fast, c2_fast = evolution_loop_fast(state.ρ, state.c₁, state.c₂, args_fast)

    @testset "State" begin
        @test norm(ρ - ρ_fast) < 1e-10
        @test tr(ρ) ≈ 1.0
        @test tr(ρ_fast) ≈ 1.0
    end
    @testset "Cavities" begin
        @test c1.length ≈ c1_fast.length
        @test c2.length ≈ c2_fast.length
        @test c1.acceleration ≈ c1_fast.acceleration
        @test c2.acceleration ≈ c2_fast.acceleration
    end

    print("Final lengths: $(c1.length) - $(c2.length)\n")

    b1 = @benchmarkable adiabatic_loop($state.ρ, $state.c₁, $state.c₂, $args)
    b2 = @benchmarkable adiabatic_loop_fast($state.ρ, $state.c₁, $state.c₂, $args_fast)
    tune!(b1)
    tune!(b2)

    display(run(b1, seconds=120))
    display(run(b2, seconds=120))

end


function make_cache(dims)


    return ()
end


function make_cache_fast(dims, forces)
    

    return (
    )
end


function evolution_loop(ρ, c1, c2, args)
    for i in range(1, 5)
        ρ, c1, c2 = meqevolve(ρ, (c1, c2), args...)
    end
    return ρ, c1, c2
end


function evolution_loop_fast(ρ, c1, c2, args)
    for i in range(1, 5)
        ρ, c1, c2 = meqevolve_fast(ρ, (c1, c2), args...)
    end
    return ρ, c1, c2
end


function masterequation(ρ, cache, ga, gb, ndims)
    function D(M, ρ)
        """Dissipator Operator appearing in the Master Equation"""
        sandwich = M * ρ * M'
        commutator = M' * M * ρ + ρ * M' * M
        return sandwich - 0.5 * commutator
    end

    cc_2ssd, cs_scp, cpcp_2sds, cpsd_sdc = cache
    first_line, second_line = Matrix{ComplexF64}(undef, ndims, ndims), Matrix{ComplexF64}(undef, ndims, ndims)
    # Dissipators
    # d_cc_2ssd = D(cc_2ssd, ρ)
    # d_cs_scp = D(cs_scp, ρ)
    first_line = 0.5 * D(cc_2ssd, ρ) + D(cs_scp, ρ)

    # d_cpcp_2sds = D(cpcp_2sds, ρ)
    # d_cpsd_sdc = D(cpsd_sdc, ρ)
    second_line = 0.5 * D(cpcp_2sds, ρ) + D(cpsd_sdc, ρ)
    
    return ga .* first_line .+ gb .* second_line
end

function masterequation_fast(ρ, cache, ga, gb, ndims)
    function D(M, ρ)
        """Dissipator Operator appearing in the Master Equation"""
        sandwich = M * ρ * M'
        commutator = M' * M * ρ + ρ * M' * M
        return sandwich - 0.5 * commutator
    end
    
    cc_2ssd, cs_scp, cpcp_2sds, cpsd_sdc, first_line, second_line = bosonic_operators
    first_line, second_line = Matrix{ComplexF64}(undef, ndims, ndims), Matrix{ComplexF64}(undef, ndims, ndims)
    # Dissipators
    # d_cc_2ssd = D(cc_2ssd, ρ)
    # d_cs_scp = D(cs_scp, ρ)
    first_line .= 0.5 .* D(cc_2ssd, ρ) .+ D(cs_scp, ρ)

    # d_cpcp_2sds = D(cpcp_2sds, ρ)
    # d_cpsd_sdc = D(cpsd_sdc, ρ)
    second_line .= 0.5 .* D(cpcp_2sds, ρ) .+ D(cpsd_sdc, ρ)
    
    return ga .* first_line .+ gb .* second_line
end
