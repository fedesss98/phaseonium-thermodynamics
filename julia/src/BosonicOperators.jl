"""
Bosonic Operators
"""
module BosonicOperators

export kraus_operators, bosonic_operators, destroy, create, C, Cp, S, Sd, pressure

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

function kraus_operators(ga, gb, Ω, Δt, ndims; sparseform=true)
    # We build the diagonal vector explicitly as ComplexF64 to prevent "Any" type errors.
    function get_diag_op(func, eigenvals)
        diag_elements = ComplexF64[func(val) for val in eigenvals]
        return spdiagm(0 => diag_elements)
    end

    # 3. Define Eigenvalues for number operators
    # For basis state |n>, n_op has val n, aa_d has val n+1
    n_vals = 0:(ndims-1)          # Eigenvalues of a'a
    aad_vals = 1:ndims            # Eigenvalues of aa' (since aa'|n> = (n+1)|n>)

    # C = cos(θ * sqrt(2 * aa'))
    C = get_diag_op(x -> cos(Ω * Δt * sqrt(2x)), aad_vals)

    # C' = cos(θ * sqrt(2 * a'a))
    Cp = get_diag_op(x -> cos(Ω * Δt * sqrt(2x)), n_vals)

    # S = a' * [sin(θ * sqrt(2 * aa')) / sqrt(2 * aa')]
    # We construct the diagonal part first, then multiply by creation operator
    S_diag = get_diag_op(x -> sin(Ω * Δt * sqrt(2x)) / sqrt(2x), aad_vals)
    a_dag = create(ndims)
    S = a_dag * S_diag

    # [cite_start]5. Construct Kraus Operators [cite: 135, 136]
    # We strictly cast scalar coefficients to ComplexF64 to ensure type stability
    I_op = _idd(ndims)
    E0 = ComplexF64(sqrt(1 - ga/2 - gb/2)) * I_op
    E1 = ComplexF64(sqrt(ga/2)) * C
    E2 = ComplexF64(sqrt(ga)) * S
    E3 = ComplexF64(sqrt(gb/2)) * Cp
    E4 = ComplexF64(sqrt(gb)) * S' # S' is S dagger
    
    if sparseform
        return [sparse(E0), sparse(E1), sparse(E2), sparse(E3), sparse(E4)]
    end
    return [E0, E1, E2, E3, E4]
end

function bosonic_operators(Ω, Δt, ndims)
    
    C = BosonicOperators.C(Ω*Δt, ndims)
    Cp = BosonicOperators.Cp(Ω*Δt, ndims)
    S = BosonicOperators.S(Ω*Δt, ndims)
    Sd = BosonicOperators.Sd(Ω*Δt, ndims)
    
    return [C, Cp, S, Sd]
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
