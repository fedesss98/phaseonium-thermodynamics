include("BosonicOperators.jl")

module OpticalCavity

export Cavity

struct Cavity
    mass::Float64
    surface::Float64
    external_force::Float64
end 

end

"""
Evolution of the System
"""
module MasterEquations

using LinearAlgebra
using ..OpticalCavity
import ..BosonicOperators

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


function meqevolve(ρ, kraus, timesteps)
    Δρ = zeros(size(ρ))
    for t in 1:timesteps
        for Ek in kraus[2:length(kraus)]
            Δρ += D(Ek, ρ)
        end
        ρ += Δρ
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


function adiabaticevolve(ρ,Δt, timesteps, cavity::Cavity)
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
        h = _free_hamiltonian(l, α0, opa, opad)
        # Evolve the System
        U = exp(-im * h)
        ρ_e = U * ρ_e * U'
        # Update pressure and acceleration
        a = pressure(ρ_e, α0, l, cavity.surface, t)*cavity.surface
        a -= cavity.external_force
        a /= cavity.mass
        #=push!(l_evolution, l)=#
    end   

    #=println(typeof(ρ_e))=#
  
    println("Cavity len evolved from $l0 to $l")
    #=cavity.length = l=#
    cavity.acceleration = a
    return ρ_e, cavity
end

end
