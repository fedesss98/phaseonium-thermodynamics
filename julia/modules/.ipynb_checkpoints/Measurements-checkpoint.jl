"""
Various measures on the Cavity
"""
module Measurements

using LinearAlgebra


"""
Gives the temperature of one quantum thermal Gibbs state
"""
function temperature(state, ω)
    real(ω / log(state[1, 1] / state[2, 2]))
end


"""
Von Neumann entropy of a density matrix, given by:
-Tr(ρ log(ρ)) = - ∑ₙλₙlog(λₙ)
where λₙare the eigenvalues of the density matrix.

Copy-pasted from qojulia/QuantumOpticsBase.jl
"""
function entropy_vn(rho; tol=1e-15) 
    evals::Vector{ComplexF64} = eigvals(rho)
    entr = zero(eltype(rho))
    for d ∈ evals
        if !(abs(d) < tol)
            entr -= d*log(d)
        end
    end
    return Float32(real(entr))
end

end

