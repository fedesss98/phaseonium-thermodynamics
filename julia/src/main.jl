includet("../modules/OpticalCavity.jl")
includet("../modules/Thermodynamics.jl")
includet("../modules/Phaseonium.jl")
includet("../modules/BosonicOperators.jl")
includet("../modules/Measurements.jl")

using LinearAlgebra
using SparseArrays
using ProgressBars
using Plots; gr()
using LaTeXStrings
using TOML
# Saving the output matrix
using Serialization
using NPZ
using Revise

using .OpticalCavity
using .Thermodynamics
using .Phaseonium
using .BosonicOperators
using .Measurements



includet("./RoutineFunctions.jl")


function print_config(io::IOStream, config)
    for (key, value) in config
        println(io, "$key: ")
        if value isa Dict
            for (k, v) in value
                println(io, "  $k: $v")
            end
        else
            println(io, "  $value")
        end
    end

end


function init(dir; config_file="")
    config = isempty(config_file) ? "/config.toml" : config_file
    try
        config = TOML.parsefile(dir * config)
    catch e
        println("Error reading configuration: $e")
    else
        println(config["meta"]["description"])
    end
    
    ndims = config["meta"]["dims"]
    global Ω = config["meta"]["omega"]
    global Δt = config["meta"]["dt"]
    
    global T_initial = config["meta"]["T1_initial"]
    
    # Find max and min frequencies
    ω_max = config["cavity1"]["alpha"] / config["cavity1"]["min_length"]
    ω_min = config["cavity1"]["alpha"] / config["cavity1"]["max_length"]
    
    # The system starts contracted, where the frequency is maximum
    ρt = thermalstate(ndims, ω_max, T_initial)
    println(
        "Initial Temperature of the Cavity: \
        $(Measurements.temperature(ρt, ω_max))")
    
    # Jump Operators
    global a = BosonicOperators.destroy(ndims)
    global ad = BosonicOperators.create(ndims)
    
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
    
    global bosonic_h = bosonic_operators(Ω, Δt, ndims)
    
    # Cooling at maximum length
    ϕ_c = π / config["phaseonium"]["phi_cold"]
    T_cold = config["phaseonium"]["T_cold"]
    α_c = Phaseonium.alpha_from_temperature(T_cold, ϕ_c, ω_min) 
    
    global ga_c, gb_c = Phaseonium.dissipationrates(α_c, ϕ_c)
    println(
        "Apparent Temperature carried by Cold Phaseonium atoms: \
        $(Phaseonium.finaltemperature(ω_min, ga_c, gb_c))")
    
    global bosonic_c = bosonic_operators(Ω, Δt, ndims);

    return config, ndims
end

function cycle_in_dir(dir)
    println("Analyzing $dir")
    config, ndims = init(dir)

    total_cycles = config["loading"]["past_cycles"]
    system_evolution = []

    # Load or create state
    state = load_or_create(dir, config)
    # Reset state
    state.ρ₁_evolution = [] 
    state.ρ₂_evolution = []
    state.c₁_evolution = []
    state.c₂_evolution = []

    # Report file
    open(dir * "/report.txt", "w") do io
        println(io, "Starting simulation at time $(now()) with config:")
        print_config(io, config)
    end
    # Visualization folder
    mkpath(dir * "/visualization")

    # Cycle
    total_cycles = 0
    isochore_time = config["stroke_time"]["isochore"]
    isochore_samplings = config["samplings"]["isochore"]
    adiabatic_time = config["stroke_time"]["adiabatic"]
    adiabatic_samplings = config["samplings"]["adiabatic"]

    total_cycle_time = isochore_samplings * 2 + 2 * adiabatic_samplings
    cycle_steps = []  # This keeps track of the total time of each stroke

    open(dir * "/report.txt", "a") do io

        state
        system_evolution
        total_cycles
        try
            for t in 1:config["meta"]["cycles"]
                println("Cycle $t")
                state, system_evolution = Main.cycle(
                    state, Δt, system_evolution, cycle_steps, 
                    isochore_time, isochore_samplings, adiabatic_time, adiabatic_samplings, io);
                total_cycles += 1
                flush(io)
            end
        catch e 
            println(io, "\n\nError '$e' at time $(now()) after $(length(system_evolution)) steps")
            println("\n\nError '$e' at time $(now()) after $(length(system_evolution)) steps")
            rethrow()
            # error("Error $e: ending cycle")
        else
            println(io, "\n\nFinished at time $(now()) after $(length(system_evolution)) steps")
            println(io, "********************** ************")
            serialize(dir * "/state_$(total_cycles)C.jl", state)
            serialize(dir * "/state_evolution_$(total_cycles)C.jl", system_evolution)
            serialize(dir * "/cavity1_evolution_$(total_cycles)C.jl", state.c₁_evolution)
            serialize(dir * "/cavity2_evolution_$(total_cycles)C.jl", state.c₂_evolution)
        end
    end
    println("Cycle len: $(size(system_evolution)[1] / total_cycles) steps")
end


function firs_fast_cycle()
    # Run a fast cycle to compile the code
    config, ndims = init(".")
    state = load_or_create('.', config)
    state.ρ₁_evolution = [] 
    state.ρ₂_evolution = []
    state.c₁_evolution = []
    state.c₂_evolution = []
    # Cycle
    isochore_time = config["stroke_time"]["isochore"]
    isochore_samplings = config["samplings"]["isochore"]
    adiabatic_time = config["stroke_time"]["adiabatic"]
    adiabatic_samplings = config["samplings"]["adiabatic"]
    system_evolution = []
    cycle_steps = []
    state, system_evolution = Main.cycle(
        state, Δt, system_evolution, cycle_steps, 
        isochore_time, isochore_samplings, adiabatic_time, adiabatic_samplings, stdout);
end

firs_fast_cycle()