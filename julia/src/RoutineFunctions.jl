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


function classic_temp(quantum_temp, ω, ϕ)
    den = 1 - quantum_temp / ω * log(1 + cos(ϕ))
    return quantum_temp / den
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

#mutable struct StrokeState{T<:Union{Real,Complex}}
#    ρ::Matrix{T}
#    c₁::Cavity
#    c₂::Cavity
#    ρ₁_evolution::Vector{Matrix{T}}
#    ρ₂_evolution::Vector{Matrix{T}}
#    c₁_evolution::Vector{Float64}
#    c₂_evolution::Vector{Float64}
#
#    StrokeState(ρ::Matrix{T}, c1::Cavity, c2::Cavity) where {T<:Union{Real,Complex}} = new{T}(ρ, c1, c2, [], [], [], [])
#    
#    StrokeState(ρ::Matrix{T}, c1::Cavity) where {T<:Union{Real,Complex}} = new{T}(ρ, c1, nothing, [], [], [], [])
#end

mutable struct StrokeState{T<:Complex}
    ρ::Matrix{T}
    c₁::Cavity
    c₂::Cavity
    ρ₁_evolution::Vector{Matrix{T}}
    ρ₂_evolution::Vector{Matrix{T}}
    c₁_evolution::Vector{Float64}
    c₂_evolution::Vector{Float64}

    StrokeState(ρ::Matrix{T}, c1::Cavity, c2::Cavity) where {T<:Complex} = new{T}(ρ, c1, c2, [], [], [], [])
    
    StrokeState(ρ::Matrix{T}, c1::Cavity) where {T<:Complex} = new{T}(ρ, c1, nothing, [], [], [], [])
end


function _check(ρ)
    println("System after the stroke:")
    if !checkdensity(ρ)
        throw(DomainError(ρ))
    end
    println("Final Temperature of the System: $(Measurements.temperature(ρ, ω))")
end

function _create_cavity(config)
    mass = config["cavity"]["mass"]
    surface = config["cavity"]["surface"]
    α0 = config["cavity"]["alpha"]
    l_min = config["cavity"]["min_length"]
    l_max = config["cavity"]["max_length"]
    expanding_force = config["cavity"]["expanding_force"]
    compressing_force = config["cavity"]["compressing_force"]
    cavity = Cavity(mass, surface, l_min, l_max, α0, expanding_force, compressing_force)
end

function load_or_create(dir, config)
    if config["loading"]["load_state"]
        filename = config["loading"]["filename"]
        cycles = config["loading"]["past_cycles"]
        println("Loading file $dir/$(filename)_$(cycles)C.jl")
        state = deserialize("$dir/$(filename)_$(cycles)C.jl")
    else
        println("Starting with a new cascade system (contracted)")
        ω = config["cavity"]["alpha"] / config["cavity"]["min_length"]
        ρt = complex(thermalstate(config["dims"], ω, config["T_initial"]))
        println("Initial Temperature of the Cavity: \
            $(Measurements.temperature(ρt, ω))")
        cavity1 = _create_cavity(config)
        cavity2 = _create_cavity(config)
        state = StrokeState(Matrix(kron(ρt, ρt)), cavity1, cavity2)
    end
    return state
end

function check_cutoff(system, ndims)
    # Jump Operators
    a = BosonicOperators.destroy(ndims)
    ad = BosonicOperators.create(ndims)
    # Check number of photons and cutoff
    ρ₁ = partial_trace(real(system), (ndims, ndims), 1)
    println("Average Photons: $(tr(ρ₁ * ad*a))")
    println("Last Element $(ρ₁[end])")
end

function bosonic_operators(Ω, Δt, ndims)
    
    C = BosonicOperators.C(Ω*Δt, ndims)
    Cp = BosonicOperators.Cp(Ω*Δt, ndims)
    S = BosonicOperators.S(Ω*Δt, ndims)
    Sd = BosonicOperators.Sd(Ω*Δt, ndims)
    
    return [C, Cp, S, Sd]
end


"""
===== CYCLE EVOLUTION =====
"""

function _phaseonium_stroke(state::StrokeState, ndims, time, bosonic, ga, gb, samplingssteps)
    stroke_evolution = Thermodynamics.phaseonium_stroke_2(
        state.ρ, time, bosonic, ga, gb; 
        sampling_steps=samplingssteps, verbose=2)

    ρ₁_evolution = [partial_trace(real(ρ), (ndims, ndims), 1) for ρ in stroke_evolution]
    ρ₂_evolution = [partial_trace(real(ρ), (ndims, ndims), 2) for ρ in stroke_evolution]
    c₁_lengths = [state.c₁.length for _ in stroke_evolution]
    c₂_lengths = [state.c₂.length for _ in stroke_evolution]
    
    append!(state.ρ₁_evolution, ρ₁_evolution)
    append!(state.ρ₂_evolution, ρ₂_evolution)
    append!(state.c₁_evolution, c₁_lengths)
    append!(state.c₂_evolution, c₂_lengths)
    
    # state.ρ = real(chop!(stroke_evolution[end]))
    state.ρ = stroke_evolution[end]
    # Jump Operators
    n = BosonicOperators.create(ndims) * BosonicOperators.destroy(ndims)
    # Print number of photons
    println("Average Photons: $(tr(state.ρ * kron(n, n)))")

    return state, stroke_evolution
end


function _adiabatic_stroke(state::StrokeState, ndims, Δt, jumps, samplingssteps)
    stroke_evolution, 
    cavity_motion, 
    total_time = Thermodynamics.adiabatic_stroke_2(
        state.ρ, [state.c₁, state.c₂], Δt, jumps;
        sampling_steps=samplingssteps, verbose=2)

    ρ₁_evolution = [partial_trace(real(ρ), (ndims, ndims), 1) for ρ in stroke_evolution]
    ρ₂_evolution = [partial_trace(real(ρ), (ndims, ndims), 2) for ρ in stroke_evolution]
    c₁_lengths = [l1 for (l1, _) in cavity_motion]
    c₂_lengths = [l2 for (_, l2) in cavity_motion]
    
    append!(state.ρ₁_evolution, ρ₁_evolution)
    append!(state.ρ₂_evolution, ρ₂_evolution)
    append!(state.c₁_evolution, c₁_lengths)
    append!(state.c₂_evolution, c₂_lengths)
    
    # state.ρ = real(chop!(stroke_evolution[end]))
    state.ρ = (stroke_evolution[end])
    state.c₁.length = cavity_motion[end][1]
    state.c₂.length = cavity_motion[end][2]
    println("$(state.c₁.compressing_force)")
    return state, stroke_evolution, total_time
end    

function cycle(state, Δt, system_evolutions, cycle_steps, isochore_t, isochore_samplings, adiabatic_t, adiabatic_samplings)
    if state isa Vector
        ρ, c₁, c₂ = state
        state = StrokeState(Matrix(ρ), c₁, c₂)
    end
    ndims = Int64(sqrt(size(state.ρ)[1]))  # Dimensions of one cavity
    
    # Isochoric Heating
    state, system_evolution = _phaseonium_stroke(state, ndims, isochore_t, bosonic_h, ga_h, gb_h, isochore_samplings)
    append!(system_evolutions, system_evolution)
    append!(cycle_steps, Δt*isochore_t)
    # Adiabatic Expansion
    state, system_evolution, adiabatic_t = _adiabatic_stroke(state, ndims, Δt, [a, ad], adiabatic_samplings)
    append!(system_evolutions, system_evolution)
    append!(cycle_steps, adiabatic_t)
    # Isochoric Cooling
    state, system_evolution = _phaseonium_stroke(state, ndims, isochore_t, bosonic_c, ga_c, gb_c, isochore_samplings)
    append!(system_evolutions, system_evolution)
    append!(cycle_steps, Δt*isochore_t)
    # Adiabatic Compression
    state, system_evolution, adiabatic_t = _adiabatic_stroke(state, ndims, Δt, [a, ad], adiabatic_samplings)
    append!(system_evolutions, system_evolution)
    append!(cycle_steps, adiabatic_t)
    
    return state, system_evolutions
end


"""
PLOTTING
"""

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


function plot_strokes_overlays(g, ys, isochore_samplings, adiabatic_samplings; x_min=0, x_max=1000)

    function _rectangle(x, w, h_up, h_down)
        Shape([
                (x, h_down),
                (x, h_up),
                (x+w, h_up),
                (x+w, h_down)
        ])
    end
    
    heating_distance = 2 * (isochore_samplings+adiabatic_samplings)
    isochore_strokes = 1:heating_distance:length(ys)
    adiabatic_strokes = isochore_samplings+adiabatic_samplings:isochore_samplings+adiabatic_samplings:length(ys)
    y_max = maximum(ys) + 0.1 * maximum(ys)
    y_min = minimum(ys) > 0 ? minimum(ys) -0.1 * minimum(ys) : minimum(ys) + 0.1 * minimum(ys) 
    for left in isochore_strokes
        plot!(g, _rectangle(left, isochore_samplings+1, y_max, y_min), fillcolor=:red, alpha=0.05, label=false)
        left_cooling = left+isochore_samplings+adiabatic_samplings+1
        plot!(g, _rectangle(left_cooling, isochore_samplings+1, y_max, y_min), fillcolor=:blue, alpha=0.05, label=false)
    end
    xlims!(x_min, x_max)
    ylims!(y_min, y_max)
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

":)"
