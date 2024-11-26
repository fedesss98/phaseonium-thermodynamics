"""
Bosonic Operators
using Base: _track_dependencies
"""
module BosonicOperators

export destroy, create, C, Cp, S, Sd, pressure

using LinearAlgebra
using SparseArrays


function destroy(ndims, sparseform=false)
    """Bosonic Annihilation operator"""
    elements = [sqrt(i) for i in 1:(ndims-1)]
    a = diagm(1 => elements)
    return sparseform ? sparse(a) : a
end


function create(ndims, sparseform=false)
    """Bosonic Creation operator"""
    a = destroy(ndims, sparseform)
    return a' 
end


function C(θ, ndims, sparseform=true)
    a = destroy(ndims)
    argument = 2 * a * a'
    operator = cos(θ * sqrt(argument))
    return sparseform ? sparse(operator) : operator
end


function Cp(θ, ndims, sparseform=true)
    a = destroy(ndims)
    argument = 2 * a' * a
    operator = cos(θ * sqrt(argument))
    return sparseform ? sparse(operator) : operator
end


function S(θ, ndims, sparseform=true)
    a = destroy(ndims)
    argument = 2 * a * a'
    sineop = sin(θ * sqrt(argument))
    dividend = pinv(sqrt(argument))  # pseudoinverse
    operator = a' * sineop * dividend
    return sparseform ? sparse(operator) : operator
end


function Sd(θ, ndims, sparseform=true)
    s = S(θ, ndims)
    operator = s'
    return sparseform ? sparse(operator) : operator
end


function kraus_operators(ρ, θ, ga, gb)
    ndims = size(ρ)

    E0 = sqrt(1 - ga/2 - gb/2) * identity(ndims)
    E1 = sqrt(ga/2) * C(θ, ndims)
    E2 = sqrt(ga) * S(θ, ndims)
    E3 = sqrt(gb/2) * Cp(θ, ndims)
    E4 = sqrt(gb) * Sd(θ, ndims)

    return [E0, E1, E2, E3, E4]
end


function _trace(A, B)
    dot(A', B)
end


function pressure(ρ, ω, v, t)
    ndims = size(ρ)
    a = destroy(ndims)
    ad = create(ndims)

    coefficient = ω / (2 v)
    expval = _trace(ad*a, ρ) 
    expval += _trace(a*ad, ρ) 
    expval -= _trace(a*a, ρ) * exp(-2 im*ω*t)
    expval -= _trace(ad*ad, ρ) * exp(2 im*ω*t)
    return coefficient * expval 
end

end
