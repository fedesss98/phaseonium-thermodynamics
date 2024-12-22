"""
Various utility functions to work with interacting Cavities-Phaseoniums
"""


using LinearAlgebra
using QuantumOptics
using SuiteSparseGraphBLAS

function thermalstate(ndims, ω, T; plotdiag::Bool=false)
    dimsrange = 1:ndims
    # Create energy eigenstates
    energies = zeros(ndims)
    for i in dimsrange
        energies[i] = exp(-ω*i/T)
    end
    energies ./= sum(energies)
    
    if plotdiag
        plt = plot(dimsrange, energies)
        display(plt)  # use display function to output 
    end

    ρ = Diagonal(energies)
end


function idd(ndims)
    Diagonal(ones(ndims))
end


function qt_ptrace(sys, trace_out)
    QuantumOptics.ptrace(sys, trace_out)
end


"""
Gives the temperature of one quantum thermal Gibbs state
"""
function temperature_of_state(state, ω)
  return ω / log(state[1, 1] / state[2, 2])
end


        
function matrixdistance(M1, M2)
    """Calculates the Frobenius distance between two matrices
    see: https://mathworld.wolfram.com/FrobeniusNorm.html"""
    return sqrt(tr((M1-M2)*(M1-M2)'))
end


function chop!(matrix, threshold = 1e-10)
    """Set small elements to zero, for real and imaginary parts separately"""
    real_parts = real.(matrix)
    imag_parts = imag.(matrix)
    
    real_parts[abs.(real_parts) .< threshold] .= 0.0
    imag_parts[abs.(imag_parts) .< threshold] .= 0.0
    
    matrix .= complex.(real_parts, imag_parts)
    return matrix
end


function isdiagonal(mat::Matrix)
    return mat == Diagonal(diag(mat))
end


function ispositive(mat::Matrix)
    return all(mat .>= 0)
end


function isreal(mat::Matrix; threshold = 1e-8)
    all(abs.(imag.(mat)) .< threshold)
end


function isnormal(mat::Matrix; threshold = 1e-5)
    return tr(mat) - 1 < threshold
end


"""Check if the last element of the matrix is really small so we can truncate the hilbert space"""
function canbecut(mat::Matrix, threshold = 1e-5)
    return last(diag(mat)) < threshold
end


function checkdensity(mat::Matrix)
    println("Real: $(isreal(mat))")
    mat = real(mat)
    println("""
        Diagonal: $(isdiagonal(mat))
        Positive: $(ispositive(mat))
        Normal: $(isnormal(mat))
        Truncatable: $(canbecut(mat))
        """)
    return ispositive(mat) && isnormal(mat) && canbecut(mat)
end

"""
Found by Copilot
"""
function partial_trace(rho::Matrix{Float64}, dims::Tuple{Int, Int}, keep::Int)
    dim1, dim2 = dims
    if keep == 1
        y = reshape(sum(reshape(rho, dim1, dim2, dim1, dim2), dims=(1, 3)), dim1, dim1)
    elseif keep == 2
        y = reshape(sum(reshape(rho, dim1, dim2, dim1, dim2), dims=(2, 4)), dim2, dim2)
    else
        throw(ArgumentError("The 'keep' argument must be 1 or 2."))
    end
    return y / tr(y)
end


mutable struct StrokeState{T<:Union{Real,Complex}}
    ρ::Matrix{T}
    c₁::Cavity
    c₂::Cavity
    ρ₁_evolution::Vector{Matrix{T}}
    ρ₂_evolution::Vector{Matrix{T}}
    c₁_evolution::Vector{Float64}
    c₂_evolution::Vector{Float64}

    StrokeState(ρ::Matrix{T}, c1::Cavity, c2::Cavity) where {T<:Union{Real,Complex}} = new{T}(ρ, c1, c2, [], [], [], [])
end
