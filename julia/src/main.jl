include("../modules/OpticalCavity.jl")
include("../modules/Thermodynamics.jl")
include("../modules/Phaseonium.jl")
include("../modules/BosonicOperators.jl")
include("../modules/Measurements.jl")

using LinearAlgebra
using SparseArrays
using ProgressBars
using Plots; gr()
using LaTeXStrings
using TOML
# Saving the output matrix
using Serialization
using NPZ

using .OpticalCavity
using .Thermodynamics
using .Phaseonium
using .BosonicOperators
using .Measurements

include("./RoutineFunctions.jl")

function init(dir)
    config = ""
    try
        config = TOML.parsefile(dir * "/config.toml")
    catch e
        println("Error reading configuration: $e")
    else
        println(config["description"])
    end
    
    NDIMS = config["dims"]
    global Ω = config["omega"]
    global Δt = config["dt"]
    
    global T_initial = config["T_initial"]
    
    # Create a Cavity
    mass = config["cavity"]["mass"]
    surface = config["cavity"]["surface"]
    α0 = config["cavity"]["alpha"]
    l_min = config["cavity"]["min_length"]
    l_max = config["cavity"]["max_length"]
    expanding_force = config["cavity"]["external_force"]
    cavity1 = Cavity(mass, surface, l_min, l_max, α0, expanding_force)
    cavity2 = Cavity(mass, surface, l_min, l_max, α0, expanding_force)
    ω_max = α0 / l_min
    ω_min = α0 / l_max
    
    # The system starts contracted, where the frequency is maximum
    ρt = thermalstate(NDIMS, ω_max, T_initial)
    println(
        "Initial Temperature of the Cavity: \
        $(Measurements.temperature(ρt, ω_max))")
    
    # Jump Operators
    global a = BosonicOperators.destroy(NDIMS)
    global ad = BosonicOperators.create(NDIMS)
    
    # Create Phaseonium atoms
    # For thermal (diagonal) phaseoniums, the Master Equation is the same as having ϕ=π/2
    # Heating at minimum length
    ϕ_h = π / config["phaseonium"]["phi_hot"]
    T_hot = config["phaseonium"]["T_hot"]
    α_h = Phaseonium.alpha_from_temperature(T_hot, ϕ_h, ω_max) 
    
    global ga_h, gb_h = Phaseonium.dissipationrates(α_h, ϕ_h)
    println(
        "Apparent Temperature carried by Hot Phaseonium atoms: \
        $(Phaseonium.finaltemperature(ω_max, ga_h, gb_h))")
    
    global bosonic_h = bosonic_operators(Ω, Δt, NDIMS)
    
    # Cooling at maximum length
    ϕ_c = π / config["phaseonium"]["phi_cold"]
    T_cold = config["phaseonium"]["T_cold"]
    α_c = Phaseonium.alpha_from_temperature(T_cold, ϕ_c, ω_min) 
    
    global ga_c, gb_c = Phaseonium.dissipationrates(α_c, ϕ_c)
    println(
        "Apparent Temperature carried by Cold Phaseonium atoms: \
        $(Phaseonium.finaltemperature(ω_min, ga_c, gb_c))")
    
    global bosonic_c = bosonic_operators(Ω, Δt, NDIMS);

    return config, NDIMS
end
