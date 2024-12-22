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
    return sandwich - 1/2 * commutator
end


function master_equation(ρ, bosonic_operators, ga, gb)
    cc_2ssd, cs_scp, cpcp_2sds, cpsd_sdc = bosonic_operators

    # Dissipators
    d_cc_2ssd = D(cc_2ssd, ρ)
    d_cs_scp = D(cs_scp, ρ)
    first_line = 0.5 * d_cc_2ssd + d_cs_scp

    d_cpcp_2sds = D(cpcp_2sds, ρ)
    d_cpsd_sdc = D(cpsd_sdc, ρ)
    second_line = 0.5 * d_cpcp_2sds + d_cpsd_sdc
    
    return ga * first_line + gb * second_line
end


function meqevolve(ρ, bosonic_operators, ga, gb, timesteps)
    # Tensor Bosonic Operators
    c, cp, s, sd = bosonic_operators

    cc_2ssd = kron(c, c) - 2 * kron(s, sd)
    cs_scp = kron(c, s) + kron(s, cp)
    cpcp_2sds = kron(cp, cp) - 2 * kron(sd, s)
    cpsd_sdc = kron(cp, sd) + kron(sd, c)
    bosonic_operators = [cc_2ssd, cs_scp, cpcp_2sds, cpsd_sdc]
    
    for _ in 1:timesteps
        ρ += master_equation(ρ, bosonic_operators, ga, gb)
    end
    return ρ
end


function krausevolve(ρ, kraus, timesteps)
    old = ρ
    dimensions = size(ρ)
    temp1 = zeros(dimensions)
    temp2 = zeros(dimensions)
    for _ in 1:timesteps
        # Reset the new density operator
        new = zeros(dimensions)
        for ek in kraus
            mul!(temp1, ek, old)  # left multiplication
            mul!(temp2, temp1, ek')  # right multiplication
            new .+= temp2
        end
        old = new  # updates the old density operator
    end
    return old
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


function adiabaticevolve(ρ, Δt, timesteps, jumps, cavity)
    ndims = size(ρ, 1)
    opa, opad = convert(Matrix{Float32}, jumps[1]), convert(Matrix{Float32}, jumps[2])

    l0 = cavity.length
    α0 = cavity.α

    a = cavity.acceleration

    global l = l0
    #=l_evolution = []=#

    ρ_e = ρ
    for t in 0:Δt:timesteps
        # Move the cavity wall
        cavity.length += 0.5 * a * Δt^2
        # Update energies
        h = _free_hamiltonian(cavity.length, α0, opa, opad)
        # Evolve the System
        U = sparse(exp(-im * h))
        ρ_e = U * ρ_e * U'
        # Update pressure and acceleration
        a = pressure(ρ_e, α0, cavity.length, cavity.surface, t)*cavity.surface
        a -= cavity.external_force
        a /= cavity.mass
        #=push!(l_evolution, l)=#
    end   
  
    #=cavity.length = l=#
    cavity.acceleration = a
    return ρ_e, cavity
end


function adiabaticevolve_2(ρ, cavities, Δt, timesteps, jumps)
    function update_a(pressure, cavity)
        (pressure * cavity.surface - cavity.external_force ) / cavity.mass
    end
    
    opa, opad = jumps
    # Reduce precision
    opa = convert(SparseMatrixCSC{Float32, Int64}, opa)
    opad = convert(SparseMatrixCSC{Float32, Int64}, opad)
    # Operators must be defined on one subspace
    dims = Int(sqrt(size(ρ)[1]))
    idd = Measurements._idd(dims)
    Δt² = Δt^2  

    # Pressure Operator
    # Decomposed in three parts (constant and rotating)
    n = opad * opa
    π_a = opa * opa
    π_ad = opad * opad

    # Preallocate variables with reduce precision
    π₁, π₂ = spzeros(ComplexF32, size(n)...), spzeros(ComplexF32, size(n)...)
    h = spzeros(Float32, size(ρ)...)
    U = spzeros(ComplexF32, size(ρ)...)
    ρ = convert(SparseMatrixCSC{ComplexF32, Int64}, ρ)  # Convert real sparse matrix to complex sparse

    c1, c2 = cavities
    α0 = c1.α

    # a1 = c1.acceleration
    # a2 = c2.acceleration
    a1 = 0  # Cavities are fixed until this time
    a2 = 0
    
    for t_idx in 0:timesteps
        t = t_idx * Δt
        # Move the cavity wall
        c1.length += 0.5 * a1 * Δt²
        c2.length += 0.5 * a2 * Δt²
        # Update energies
        ω₁ = α0 / c1.length
        ω₂ = α0 / c2.length
        h1 = _free_hamiltonian(ω₁, n)
        h2 = _free_hamiltonian(ω₂, n)
        kron!(h, h1, h2)
        # Evolve the System
        U .= padm(-im .* h)
        mul!(ρ, U, ρ)
        mul!(ρ, ρ, U')
        # Update pressure and acceleration
        @. π₁ = (2 * n + 1)  - (π_a * exp(-2*im*ω₁*t)) - (π_ad * exp(2*im*ω₁*t))
        @. π₂ = (2 * n + 1)  - (π_a * exp(-2*im*ω₂*t)) - (π_ad * exp(2*im*ω₂*t))
        println(typeof(π₂))
        p1 = Measurements.pressure(ρ, π₁, idd, α0, c1.length, c1.surface; s=1)
        p2 = Measurements.pressure(ρ, π₂, idd, α0, c2.length, c2.surface; s=2)
        a1 = (p1 * c1.surface - c1.external_force) / c1.mass
        a2 = (p2 * c2.surface - c2.external_force) / c2.mass
    end
  
    c1.acceleration = a1
    c2.acceleration = a2

    return ρ, c1, c2
end

end

using BenchmarkTools
using LinearAlgebra
using Expokit
using .OpticalCavity
using .BosonicOperators
using .MasterEquations
include("../src/RoutineFunctions.jl")


function benchmark_adiabatic_stroke(ρ, cavities, Δt, timesteps, jumps)
    result = MasterEquations.adiabaticevolve_2(ρ, cavities, Δt, timesteps, jumps)
    return 0
end


println("MAIN")
function stroke_params(time)
    ndims = 20
    timesteps = time 
    Δt = 1e-2

    ρt = thermalstate(ndims, 1.0, 1.5)
    ρ = sparse(kron(ρt, ρt))

    c = OpticalCavity.Cavity(1.0, 1.0, 1.0, π, 0.0, 0.5)
    cavities = [c, c]

    a = BosonicOperators.destroy(ndims, true)
    ad = BosonicOperators.create(ndims, true)
    jumps = [a, ad]
    
    return ρ, cavities, Δt, timesteps, jumps
end

#=@benchmark benchmark_adiabatic_stroke($stroke_params()...)=#

