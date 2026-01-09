"""
Various utility functions to work with interacting Cavities-Phaseoniums
"""

using LinearAlgebra

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
    den = 1 - (quantum_temp / ω) * log(1 + cos(ϕ))
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
function partial_trace(rho::Matrix{T}, dims::Tuple{Int, Int}, keep::Int) where {T<:Union{Real, Complex}}
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



function _check(ρ)
    println("System after the stroke:")
    if !checkdensity(ρ)
        throw(DomainError(ρ))
    end
    println("Final Temperature of the System: $(Measurements.temperature(ρ, ω))")
end

function _create_cavity(cavity_config)
    mass = cavity_config["mass"]
    surface = cavity_config["surface"]
    α0 = cavity_config["alpha"]
    γ = get(cavity_config, "friction", 0.0)
    l_min = cavity_config["min_length"]
    l_max = cavity_config["max_length"]
    expanding_force = cavity_config["expanding_force"]
    compressing_force = cavity_config["compressing_force"]
    cavity = Cavity(mass, surface, l_min, l_max, α0, γ, expanding_force, compressing_force)
    return cavity
end

"""
    load_or_create(dir, config)
Create a Density Matrix based on `config` file.
To create a one-cavity system, set `ω2` to zero,
else it will create a two-cavity product state with specified params.
If the `loading` section of the `config` file is provided,
it will try to load a state from a previous run.

# Returns
- `state`: a StrokeState struct containing the system Density Matrix and its evolution history.
"""
function load_or_create(dir, config)
    if config["loading"]["load_state"]
        filename = config["loading"]["filename"]
        cycles = config["loading"]["past_cycles"]
        println("Loading file $dir/$(filename)_$(cycles)C.jl")
        state = deserialize("$dir/$(filename)_$(cycles)C.jl")
    else
        # println("Starting with a new cascade system (contracted)")
        ω1 = config["cavity1"]["alpha"] / config["cavity1"]["min_length"]
        ρt1 = complex(thermalstate(config["meta"]["dims"], ω1, config["meta"]["T1_initial"]))
        cavity1 = _create_cavity(config["cavity1"])
        ω2 = config["cavity2"]["alpha"] / config["cavity2"]["min_length"]
        if ω2 == 0 
            # println("Initial Temperature of the Cavity: \
            #   $(Measurements.temperature(ρt1, ω1))")
            state = StrokeState(Matrix(ρt1), cavity1)
        else
            ρt2 = complex(thermalstate(config["meta"]["dims"], ω2, config["meta"]["T2_initial"]))
            cavity2 = _create_cavity(config["cavity2"])
            # println("Initial Temperature of the Cavities: \
            #     $(Measurements.temperature(ρt1, ω1)) - $(Measurements.temperature(ρt2, ω2))")
            state = StrokeState(Matrix(kron(ρt1, ρt2)), cavity1, cavity2)
        end
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

"""
==========
PHASEONIUM
==========
"""


"""
Gives the rates of dissipation appearing in the Phaseonium Master Equation
"""
function dissipationrates(α, ϕ)
    ga = 2*α^2
    gb = (1 + cos(ϕ))*(1 - α^2)
    return real(ga), real(gb)
end


"""
Gives the stable apparent temperature carried by Phaseonium atoms
"""
function finaltemperature(ω, γα, γβ)
    return - ω / log(γα/γβ)
end


"""
Find the parameter ϕ that gives the Apparent Temperature specified, given α
"""
function phi_from_temperature(T, α, ω)
    T = T * ω
    return arccos(2*α^2 * exp(1/T) / (1-α^2) - 1)
end

"""
Find the parameter α that gives the Apparent Temperature specified, given ϕ
"""
function alpha_from_temperature(T, ϕ, ω)
    factor = (1+cos(ϕ)) / (2 * exp(ω/T) + 1 + cos(ϕ)) 
    return sqrt(factor)
end
