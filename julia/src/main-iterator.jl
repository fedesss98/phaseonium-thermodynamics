using MKL
using Revise
using Dates
using CSV
using DataFrames
using LinearAlgebra
using SparseArrays
using ProgressBars
using Plots
using LaTeXStrings
using TOML
# Saving the output matrix
using Serialization
using NPZ

includet("../modules/OpticalCavity.jl")
includet("../modules/Thermodynamics.jl")
includet("../modules/MasterEquations.jl")
includet("../modules/Phaseonium.jl")
includet("../modules/BosonicOperators.jl")
includet("../modules/Measurements.jl")


using .OpticalCavity
using .Thermodynamics
using .MasterEquations
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
    config = isempty(config_file) ? "/fast_config.toml" : config_file
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
    # println(
    #     "Initial Temperature of the Cavity: \
    #     $(Measurements.temperature(ρt, ω_max))")
    
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
    # println(
    #     "Apparent Temperature carried by Hot Phaseonium atoms: \
    #     $(Phaseonium.finaltemperature(ω_max, ga_h, gb_h))")
    
    global bosonic_h = bosonic_operators(Ω, Δt, ndims)
    
    # Cooling at maximum length
    ϕ_c = π / config["phaseonium"]["phi_cold"]
    T_cold = config["phaseonium"]["T_cold"]
    α_c = Phaseonium.alpha_from_temperature(T_cold, ϕ_c, ω_min) 
    
    global ga_c, gb_c = Phaseonium.dissipationrates(α_c, ϕ_c)
    # println(
    #     "Apparent Temperature carried by Cold Phaseonium atoms: \
    #     $(Phaseonium.finaltemperature(ω_min, ga_c, gb_c))")
    
    global bosonic_c = bosonic_operators(Ω, Δt, ndims);

    return config, ndims
end


function run_simulation(state, config, io, dir)
    # Cycle
    total_cycles = 0
    system_evolution = []
    Δt = config["meta"]["dt"]
    isochore_time = config["stroke_time"]["isochore"]
    isochore_samplings = config["samplings"]["isochore"]
    adiabatic_time = config["stroke_time"]["adiabatic"]
    adiabatic_samplings = config["samplings"]["adiabatic"]

    total_cycle_time = isochore_samplings * 2 + 2 * adiabatic_samplings
    cycle_steps = []  # This keeps track of the total time of each stroke
    error = 0

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
        # error("Error $e: ending cycle")
        error = e
    else
        println(io, "\n\nFinished at time $(now()) after $(length(system_evolution)) steps")
        println(io, "********************** ************")
        serialize(dir * "/state_$(total_cycles)C.jl", state)
        serialize(dir * "/state_evolution_$(total_cycles)C.jl", system_evolution)
        serialize(dir * "/cavity1_evolution_$(total_cycles)C.jl", state.c₁_evolution)
        serialize(dir * "/cavity2_evolution_$(total_cycles)C.jl", state.c₂_evolution)
    end
    return state, system_evolution, total_cycle_time, cycle_steps, error
end


function save_last_cycle(dir, config, system_evolution, state, total_cycle_time, cycle_steps)
    α0 = config["cavity1"]["alpha"]

    last_evolution = system_evolution[end-total_cycle_time+1:end]
    ω₁_evolution = [α0 / l1 for l1 in state.c₁_evolution[end-total_cycle_time+1:end]]
    if length(state.c₂_evolution) > 0
        ω₂_evolution = [α0 / l2 for l2 in state.c₂_evolution[end-total_cycle_time+1:end]]
    else
        ω₂_evolution = 0.0
    end

    dict_evolution = Dict(string(i)=>last_evolution[i] for i in eachindex(last_evolution))
    cycle_steps = Vector{Float64}(cycle_steps)

    npzwrite(dir * "/cycle_steps.npy", cycle_steps[end-3:end])
    npzwrite(dir * "/cavities_evolution_1cycle.npz", c1=ω₁_evolution, c2=ω₂_evolution)
    npzwrite(dir * "/cascade_evolution_1cycle.npz", dict_evolution)
end


function cycle_in_dir(dir)
    # Check if report.txt already exists
    if isfile(dir * "/report.txt")
        return
    end

    # Trace distance: 1/2 * Tr(|ρ1 - ρ2|)
    # to see if the cycle converged
    function trace_distance(ρ1, ρ2)
        diff = ρ1 - ρ2
        # Calculate eigenvalues of the difference
        eigenvals = eigvals(diff)
        # Sum of absolute values of eigenvalues, divided by 2
        return sum(abs.(eigenvals)) / 2
    end

    println("\n\n_____________________________")
    config, ndims = init(dir)
    
    # Load or create state
    state = load_or_create(dir, config)
    # Reset state
    state.ρ₁_evolution = [] 
    state.ρ₂_evolution = []
    state.c₁_evolution = []
    state.c₂_evolution = []
    # Initialize variables
    system_evolution = []
    total_cycle_time = 0
    cycle_steps = []
    e = 0

    # Create Report file
    open(dir * "/report.txt", "w") do io
        println(io, "Starting simulation at time $(now()) with config:")
        print_config(io, config)
    end
    # Create Visualization folder
    mkpath(dir * "/visualization")

    open(dir * "/report.txt", "a") do io
        println("Expanding Force: $(config["cavity1"]["expanding_force"])")
        println("Compressing Force: $(config["cavity1"]["compressing_force"])")
        sim_results = run_simulation(state, config, io, dir)
        state, system_evolution, total_cycle_time, cycle_steps, e = sim_results
        if e == 0
            ρᵢ_state_last_cycle = state.ρ₁_evolution[end-total_cycle_time]
            ρₑ_state_last_cycle = state.ρ₁_evolution[end]
            distance = trace_distance(ρᵢ_state_last_cycle, ρₑ_state_last_cycle)
            println(io, "Trace distance between the initial and final state of last cycle: $distance")
            println("Trace distance between the initial and final state of last cycle: $distance")
            save_last_cycle(dir, config, system_evolution, state, total_cycle_time, cycle_steps)
        else
            println("Simulation returned the error '$e'.")
        end
    end
end


function firs_fast_cycle()
    # FIRST FAST CYCLE to compile code
    # Throw out the output
    original_stdout = stdout
    redirect_stdout(devnull)
    try
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
    finally
        redirect_stdout(original_stdout)
    end
end

function main_iterator()
    println("New version created $(now())")
    CSV_FILE = "./simulations/simulations_ledger.csv";

    csv = CSV.read(CSV_FILE, DataFrame, header=1);
    dirs = ["simulations/$n" for n in csv[!, "meta_name"]];

    println("There are $(nrow(csv)) simulation files.")

    for dir in dirs
        #print(dir)
        cycle_in_dir(dir)
    end

    println("\n\n---------------\n\nFinished!!")
end


println("Running a fast cycle to compile the file")
firs_fast_cycle()
println("Done.")

# main_iterator()

