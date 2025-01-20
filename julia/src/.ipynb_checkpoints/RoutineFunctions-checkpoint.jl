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

function coherentstate(alpha, dim)
    # Create the coherent state vector
    psi = zeros(ComplexF64, dim)
    for n in 0:(dim-1)
        if n <= 20
            psi[n+1] = exp(-0.5*abs2(alpha)) * (alpha^n) / sqrt(factorial(n))
        else
            psi[n+1] = 0
        end
    end

    # Create the density matrix
    ρ = psi * psi'
    return ρ
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


function chop!(matrix; threshold = 1e-10)
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

function is_gaussian_state(ρ, a, ad)
    # Define the quadrature operators
    x = (a + ad) / sqrt(2)
    p = (a - ad) / (im * sqrt(2))

    # Calculate the first moments (mean values)
    mean_values = [tr(x * ρ), tr(p * ρ)]

    # Calculate the covariance matrix
    covariance_matrix = [
        tr((x*x + x*x) * ρ) - 2*mean_values[1]*mean_values[1] tr((x*p + p*x) * ρ) - 2*mean_values[1]*mean_values[2];
        tr((p*x + x*p) * ρ) - 2*mean_values[2]*mean_values[1] tr((p*p + p*p) * ρ) - 2*mean_values[2]*mean_values[2]
    ]
    # covariance_matrix = chop!(covariance_matrix)
    @. covariance_matrix = round(covariance_matrix)
    # Check if the covariance matrix is positive semi-definite
    is_positive_semi_definite = all(real(eigvals(covariance_matrix)) .>= 0)

    # Check if the covariance matrix satisfies the uncertainty principle
    satisfies_uncertainty_principle = real(det(covariance_matrix)) >= 1

    return is_positive_semi_definite && satisfies_uncertainty_principle
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
    
    StrokeState(ρ::Matrix{T}, c1::Cavity) where {T<:Union{Real,Complex}} = new{T}(ρ, c1, nothing, [], [], [], [])
end

function _check(ρ)
    println("System after the stroke:")
    if !checkdensity(ρ)
        throw(DomainError(ρ))
    end
    println("Final Temperature of the System: $(Measurements.temperature(ρ, ω))")
end

function measure_and_plot(x, y, system_evolution, cavity_evolution, title; α=π)
    ys = []
    xs = []
    if x == "Entropy"
        x_measurement = Measurements.entropy_vn
        x_label = x
    elseif x == "Frequency"
        _frequency(ρ, ω) = ω
        x_measurement = _frequency
        x_label = L"\omega"
    end

    if y == "Energy"
        y_measurment = Measurements.avg_E
        y_label = y
    elseif y == "n"
        y_measurment = Measurements.avg_number
        y_label = L"\langle \hat{n} \rangle"
    elseif y == "Temperature"
        y_measurment = Measurements.temperature
        y_label = y
    end
    
    for (i, ρ) in enumerate(system_evolution)
        cavity_len = cavity_evolution isa Real ? cavity_evolution : cavity_evolution[i]
        local ω = α / cavity_len
        x = real(round(x_measurement(ρ, ω), digits=5))
        y = real(round(y_measurment(ρ, ω), digits=5))
        
        push!(xs, x)
        push!(ys, y)
    end

    g = plot(xs, ys, label="Stroke")
        
    # Plot starting point
    scatter!(g, [xs[1]], [ys[1]], label="Start", mc="blue", ms=5, msw=0)
    # Plot ending point
    scatter!(g, [xs[end]], [ys[end]], label="End", mc="red", ms=2.5, msw=0)
    title!(title)
    xlabel!(x_label)
    ylabel!(y_label)
    
    return g
end

function plot_strokes_overlays(g, temperatures, isochore_samplings, adiabatic_samplings; x_max=1000)

    function rectangle(x, w, h)
        Shape([
                (x, 0),
                (x, h),
                (x+w, h),
                (x+w, 0)
        ])
    end
    
    heating_distance = 2 * (isochore_samplings+adiabatic_samplings) + 4
    isochore_strokes = 1:heating_distance:length(temperatures)
    adiabatic_strokes = isochore_samplings+3+adiabatic_samplings:isochore_samplings+adiabatic_samplings:length(temperatures)
    y_max = maximum(temperatures) + 0.1 * maximum(temperatures)
    for left in isochore_strokes
        plot!(g, rectangle(left, isochore_samplings+1, y_max), fillcolor=:red, alpha=0.05, label=false)
        left_cooling = left+isochore_samplings+adiabatic_samplings+2
        plot!(g, rectangle(left_cooling, isochore_samplings+1, y_max), fillcolor=:blue, alpha=0.05, label=false)
    end
    xlims!(0, x_max)
    ylims!(0, y_max)
end

function plot_in_time(observable, system_evolution, cavity_evolution, label, title; 
        g=nothing, α=π, isochore_samplings=1, adiabatic_samplings=1, x_max=1000)
    temperatures = []
    if observable == "n"
        measurement = Measurements.avg_number
        y_label = L"\langle \hat{n} \rangle"
    elseif observable == "T"
        measurement = Measurements.temperature
        y_label = "Temperature"
    end
    
    for (i, ρ) in enumerate(system_evolution)
        cavity_len = cavity_evolution isa Real ? cavity_evolution : cavity_evolution[i]
        local ω = α / cavity_len
        t = real(round(measurement(ρ, ω), digits=5))
        push!(temperatures, t)
    end

    if isnothing(g)
        # Compose the whole plot with overlayed strokes
        g = plot(temperatures, label=label)
        plot_strokes_overlays(g, temperatures, isochore_samplings, adiabatic_samplings)
    else
        plot!(temperatures, label=label)
    end
    
    title!(title)
    xlabel!("Time")
    ylabel!(y_label)

    return g
    
end
