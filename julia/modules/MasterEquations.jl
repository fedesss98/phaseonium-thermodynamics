include("BosonicOperators.jl")
include("Measurements.jl")
include("OpticalCavity.jl")

"""
Evolution of the System
"""
module MasterEquations

using LinearAlgebra
using SparseArrays
using Expokit

using ..OpticalCavity
using ..BosonicOperators
using ..Measurements

export meqevolve, krausevolve

"""
Implements the Dissipator superoperator
"""
function D(M, ρ)
    """Dissipator Operator appearing in the Master Equation"""
    sandwich = M * ρ * M'
    commutator = M' * M * ρ + ρ * M' * M
    return sandwich - 0.5 * commutator
end


function master_equation_2(ρ, bosonic_operators, ga, gb, ndims)
    cc_2ssd, cs_scp, cpcp_2sds, cpsd_sdc = bosonic_operators
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


function meqevolve_2(ρ, bosonic_operators, ga, gb, timesteps, ndims)
    
    for _ in 1:timesteps
        ρ += master_equation_2(ρ, bosonic_operators, ga, gb, ndims)
    end
    return ρ
end

function master_equation(ρ, bosonic_operators, ga, gb, ndims)
    c, cp, s, sd = bosonic_operators
    first_line, second_line = Matrix{ComplexF64}(undef, ndims, ndims), Matrix{ComplexF64}(undef, ndims, ndims)
    # Dissipators
    first_line = 0.5 * D(c, ρ) + D(sd, ρ)

    second_line = 0.5 * D(cp, ρ) + D(s, ρ)
    
    return ga .* first_line .+ gb .* second_line
end


function meqevolve(ρ, bosonic_operators, ga, gb, timesteps, ndims)
    
    for _ in 1:timesteps
        ρ += master_equation(ρ, bosonic_operators, ga, gb, ndims)
    end
    return ρ
end


function krausevolve(ρ, kraus, timesteps)
    dimensions = size(ρ)
    temp1 = spzeros(ComplexF64, dimensions)
    for _ in 1:timesteps
        temp2 = spzeros(ComplexF64, dimensions)
        for ek in kraus
            mul!(temp1, ek, ρ)  # left multiplication
            mul!(temp1, temp1, ek')  # right multiplication
            temp2 .+= temp1
        end
        ρ = temp2  # updates the old density operator
    end
    return ρ
end


function approxevolve(ρ, jumps, rates, timesteps)
    old = ρ
    a, ad = jumps
    ga, gb = rates
    Δρ = zeros(size(ρ))
    for i in 1:timesteps
        Δρ += ga*D(ad, ρ)
        Δρ += gb*D(a,ρ)
        ρ += Δρ
    end

end


function _free_hamiltonian(ω, n)
   0.5 * ω * n
end


function adiabaticevolve(ρ, Δt, timesteps, jumps, cavity, idd, π_parts)
    ndims = size(ρ, 1)

    n, π_a, π_ad = π_parts
    opa, opad = convert(Matrix{Float32}, jumps[1]), convert(Matrix{Float32}, jumps[2])

    l0 = cavity.length
    α0 = cavity.α

    a = cavity.acceleration
    
    for t in 0:Δt:timesteps
        # Move the cavity wall
        cavity.length += 0.5 * a * Δt^2
        ω = α0 / cavity.length
        # Update energies
        h = 0.5 * ω .* n
        # Evolve the System
        U = padm(-im .* h)
        ρ = U * ρ * U'
        # Update pressure and acceleration
        π_op = (2 * n + idd)  - (π_a * exp(-2*im*ω*t)) - (π_ad * exp(2*im*ω*t))
        p = Measurements.pressure(ρ, π_op, α0, cavity.length, cavity.surface)
        a = (p * cavity.surface - cavity.external_force) / cavity.mass
    end   
  
    cavity.acceleration = a
    return ρ, cavity
end


function _adiabaticevolve_2(ρ, cavities, Δt, t, timesteps, allocated_op, π_parts)
    U, idd = allocated_op
    n, π_a, π_ad = π_parts
    
    Δt² = Δt^2  

    c1, c2 = cavities
    α0 = c1.α

    a1 = c1.acceleration
    a2 = c2.acceleration
    
    for t_idx in timesteps*(t-1)+1:timesteps*t
        t = t_idx * Δt
        # Move the cavity wall
        c1.length += 0.5 * a1 * Δt²
        c2.length += 0.5 * a2 * Δt²
        # Update energies
        ω₁ = α0 / c1.length
        ω₂ = α0 / c2.length
        h1 = 0.5 * ω₁ .* n
        h2 = 0.5 * ω₂ .* n
        h = kron(h1, h2)
        # Evolve the System
        U .= padm(-im .* h)
        ρ = U * ρ * U'
        # Update pressure and acceleration
        π₁ = (2 * n + idd)  - (π_a * exp(-2*im*ω₁*t)) - (π_ad * exp(2*im*ω₁*t))
        π₂ = (2 * n + idd)  - (π_a * exp(-2*im*ω₂*t)) - (π_ad * exp(2*im*ω₂*t))
        p1 = Measurements.pressure(ρ, π₁, idd, α0, c1.length, c1.surface; s=1)
        p2 = Measurements.pressure(ρ, π₂, idd, α0, c2.length, c2.surface; s=2)
        a1 = (p1 * c1.surface - c1.external_force) / c1.mass
        a2 = (p2 * c2.surface - c2.external_force) / c2.mass
    end

    c1.acceleration = a1
    c2.acceleration = a2

    return ρ, c1, c2
end


function adiabaticevolve_2(ρ, cavities, Δt, t, allocated_op, π_parts, process, stop1, stop2)
    U, idd = allocated_op
    n, π_a, π_ad = π_parts
    
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
    h1 = 0.5 * ω₁ .* n
    h2 = 0.5 * ω₂ .* n
    h = kron(h1, h2)
    
    # Evolve the System
    U .= padm(-im .* h)
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
    
    println("p1:$p1 - p2:$p2\nf1:$force1/$a1 - f2:$force2/$a2")
    
    if norm(a1) <= 0.01 || norm(a2) <= 0.01
        error(
            "One cavity is almost still during $process \
            with force $force1/$force2 and pressure $p1/$p2")
    end
    if process == "Expansion" && (a1 < 0 || a2 < 0)
        error("One cavity is going backward during expansion!")
    elseif process == "Contraction" && (a1 > 0 || a2 > 0)
        error("One cavity is going forward during contraction!")
    end
    if c1.acceleration * a1 < 0 || c2.acceleration * a2 < 0
        error("One cavity changed direction during $process!")
    end
    c1.acceleration = a1
    c2.acceleration = a2

    return ρ, c1, c2
end

end

#=using BenchmarkTools=#
#=using LinearAlgebra=#
#=using Expokit=#
#=using .OpticalCavity=#
#=using .BosonicOperators=#
#=using .MasterEquations=#
#=include("../src/RoutineFunctions.jl")=#
#==#
#==#
#=function benchmark_adiabatic_stroke(ρ, cavities, Δt, timesteps, jumps)=#
#=    result = MasterEquations.adiabaticevolve_2(ρ, cavities, Δt, timesteps, jumps)=#
#=    return 0=#
#=end=#
#==#
#==#
#=println("MAIN")=#
#=function stroke_params(time)=#
#=    ndims = 20=#
#=    timesteps = time =#
#=    Δt = 1e-2=#
#==#
#=    ρt = thermalstate(ndims, 1.0, 1.5)=#
#=    ρ = sparse(kron(ρt, ρt))=#
#==#
#=    c = OpticalCavity.Cavity(1.0, 1.0, 1.0, π, 0.0, 0.5)=#
#=    cavities = [c, c]=#
#==#
#=    a = BosonicOperators.destroy(ndims, true)=#
#=    ad = BosonicOperators.create(ndims, true)=#
#=    jumps = [a, ad]=#
#==#
    #=return ρ, cavities, Δt, timesteps, jumps=#
#=end=#

#=@benchmark benchmark_adiabatic_stroke($stroke_params()...)=#

