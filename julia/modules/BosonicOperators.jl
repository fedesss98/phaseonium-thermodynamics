"""
Bosonic Operators
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


function _idd(ndims)
    Diagonal(ones(ndims))
end


function pressure(ρ, π, idd, α, l, S; s=0)

    coefficient = α / (2*l^2*S)

    ω = α / l
    
    # Pressure Operator
    if s == 1
        op = kron(π, idd)
    elseif s == 2
        op = kron(idd, π)
    else
        op = π
    end

    real(coefficient * _trace(ρ, op))
end

end
