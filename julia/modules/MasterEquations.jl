include("BosonicOperators.jl")

include("OpticalCavity.jl")

"""
Evolution of the System
"""
module MasterEquations

using LinearAlgebra
using ..OpticalCavity
using ..BosonicOperators

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


function _free_hamiltonian(l, α0, a, ad)
   1/2 * α0/l * ad*a
end


function adiabaticevolve(ρ, Δt, timesteps, jumps, cavity)
    ndims = size(ρ, 1)
    opa, opad = jumps

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
        U = exp(-im * h)
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

    c1, c2 = cavities
    α0 = c1.α

    # a1 = c1.acceleration
    # a2 = c2.acceleration
    a1 = 0  # Cavities are fixed until this time
    a2 = 0
    
    for t in 0:Δt:timesteps
        # Move the cavity wall
        c1.length += 0.5 * a1 * Δt^2
        c2.length += 0.5 * a2 * Δt^2
        # Update energies
        h1 = _free_hamiltonian(c1.length, α0, opa, opad)
        h2 = _free_hamiltonian(c2.length, α0, opa, opad)
        h = kron(h1, h2)
        # Evolve the System
        U = exp(-im * h)
        ρ = U * ρ * U'
        # Update pressure and acceleration
        p1 = pressure(ρ, α0, c1.length, c1.surface, t)
        p2 = pressure(ρ, α0, c2.length, c2.surface, t)
        a1 = update_a(p1, c1)
        a2 = update_a(p2, c2)
    end   
  
    c1.acceleration = a1
    c2.acceleration = a2
    return ρ, c1, c2
end

end
