"""
Various measures on the Cavity
"""
module Measurements

using LinearAlgebra


"""
Gives the temperature of one quantum thermal Gibbs state
"""
function temperature(state, ω, args...)
    real(ω / log(state[1, 1] / state[2, 2]))
end


"""
Von Neumann entropy of a density matrix, given by:
-Tr(ρ log(ρ)) = - ∑ₙλₙlog(λₙ)
where λₙare the eigenvalues of the density matrix.

Copy-pasted from qojulia/QuantumOpticsBase.jl
"""
function entropy_vn(rho, args...; tol=1e-15) 
    evals::Vector{ComplexF64} = eigvals(rho)
    entr = zero(eltype(rho))
    for d ∈ evals
        if !(abs(d) < tol)
            entr -= d*log(d)
        end
    end
    return Float32(real(entr))
end


"""
Gives the expected value of photon's number in the Cavity
"""
function avg_number(state, ω, args...)
    dims = size(state)[1]
    N = Diagonal(0:dims-1)

    return tr(state * N)
end

"""
Gives the expected value of the energy of the cavity
"""
function avg_E(state, ω, args...)
    dims = size(state)[1]
    N = Diagonal(0:dims-1)
    
    return 0.5 * ω * tr(state * N)
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

    real(coefficient * tr(ρ * op))
end


function pressure(ρ, π, α, l, S)

    coefficient = α / (2*l^2*S)

    ω = α / l

    real(coefficient * tr(ρ * π))
end

end

