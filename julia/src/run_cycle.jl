"""
Code to run the cycle in a folder with arguments
"""

include("../modules/OpticalCavity.jl")
include("../modules/Thermodynamics.jl")
include("../modules/Phaseonium.jl")
include("../modules/BosonicOperators.jl")
include("../modules/Measurements.jl")

using Configurations
using LinearAlgebra
using ProgressBars
using Plots
using LaTeXStrings
# Saving the output matrix
using Serialization
using NPZ

using .OpticalCavity
using .Thermodynamics
using .Phaseonium
using .BosonicOperators
using .Measurements

include("./RoutineFunctions.jl")

@option struct PhaseoniumOptions
    phi_hot::Float64 = 0.0
    T_hot::Float64 = 0.0
    phi_cold::Float64 = 0.0
    T_cold::Float64 = 0.0
end

@option struct CavityOptions
    mass::Float64 = 0.0
    surface::Float64 = 0.0
    length::Float64 = 0.0
    alpha::Float64 = 0.0
    acceleration::Float64 = 0.0
    external_force::Float64 = 0.0
end

@option struct StrokesOptions
    isochore::Int = 0
    adiabatic::Int = 0
end

@option struct SamplingOptions
    isochore::Int = 0
    adiabatic::Int = 0
end

@option struct LoadingOptions
    load_state::Bool = false
    filename::String = ""
    past_cycles::Int = 0
end

@option struct SimOptions
    description::String
    dims::Int
    omega::Float64
    dt::Float64
    cycles::Int
    T_initial::Float64

    phaseonium::PhaseoniumOptions = PhaseoniumOptions()
    cavity::CavityOptions = CavityOptions()
    stroke_time::StrokesOptions = StrokesOptions()
    samplings::SamplingOptions = SamplingOptions()
    loading::LoadingOptions = LoadingOptions()
end
    

function read_configuration(dir)
    from_toml(SimOptions, dir * "/config.toml")
end

function create_cavity(cavity_options)
    m = cavity_options.mass
    s = cavity_options.surface
    l = cavity_options.length
    alpha = cavity_options.alpha
    a = cavity_options.acceleration
    f = cavity_options.external_force
    Cavity(m, s, l, alpha, a, f)
end

function create_joint_system(options, ω)
    ρt = thermalstate(options.dims, ω, options.T_initial)
    println(
        "Initial Temperature of the Cavity:
        $(Measurements.temperature(ρt, ω))")
    # Joint system
    ρ_tot = kron(ρt, ρt)
end

function bosonic_operators(α, ϕ, Ω, Δt, ndims)
    
    C = BosonicOperators.C(Ω*Δt, ndims)
    Cp = BosonicOperators.Cp(Ω*Δt, ndims)
    S = BosonicOperators.S(Ω*Δt, ndims)
    Sd = BosonicOperators.Sd(Ω*Δt, ndims)
    
    return [C, Cp, S, Sd]
end

function create_phaseoniums(options, ω)
    # Heating
    ϕ_h = π / options.phaseonium.phi_hot
    α_h = Phaseonium.alpha_from_temperature(options.phaseonium.T_hot, ϕ_h, ω)
    
    
    # Cooling
    ϕ_c = π / options.phaseonium.phi_cold
    α_c = Phaseonium.alpha_from_temperature(options.phaseonium.T_cold, ϕ_c, ω) 
    
    ga_h, gb_h = Phaseonium.dissipationrates(α_h, ϕ_h)
    ga_c, gb_c = Phaseonium.dissipationrates(α_c, ϕ_c)
    
    println(
        "Apparent Temperature carried by Hot Phaseonium atoms: 
        $(Phaseonium.finaltemperature(ω, ga_h, gb_h))")
    println(
        "Apparent Temperature carried by Cold Phaseonium atoms: 
        $(Phaseonium.finaltemperature(ω, ga_c, gb_c))")
    
    bosonic_h = bosonic_operators(α_h, ϕ_h, options.omega, options.dt, options.dims)
    bosonic_c = bosonic_operators(α_c, ϕ_c, options.omega, options.dt, options.dims)
    
    return (ga_h, gb_h, bosonic_h), (ga_c, gb_c, bosonic_c)
end

function load_or_create_state(options, ρ, cavity1, cavity2; dir=ARGS[1])
    if options.load_state
        return deserialize(dir * "/" * options.filename), options.past_cycles
    else
        return StrokeState(Matrix(ρ), cavity1, cavity2), 0
    end
end


function _phaseonium_stroke(state::StrokeState, time, bosonic, ga, gb, ss)
    stroke_evolution = Thermodynamics.phaseonium_stroke_2(
        state.ρ, time, bosonic, ga, gb; 
        sampling_steps=ss, verbose=1)

    dims = Int(sqrt(size(state.ρ)[1]))
    ρ₁_evolution = [partial_trace(real(ρ), (dims, dims), 1) for ρ in stroke_evolution]
    ρ₂_evolution = [partial_trace(real(ρ), (dims, dims), 2) for ρ in stroke_evolution]
    c₁_lengths = [state.c₁.length for _ in stroke_evolution]
    c₂_lengths = [state.c₂.length for _ in stroke_evolution]

    append!(state.ρ₁_evolution, ρ₁_evolution)
    append!(state.ρ₂_evolution, ρ₂_evolution)
    append!(state.c₁_evolution, c₁_lengths)
    append!(state.c₂_evolution, c₂_lengths)
    
    state.ρ = real(chop!(stroke_evolution[end]))
    return state, stroke_evolution
end


function _adiabatic_stroke(state::StrokeState, time, Δt, jumps, ss)
    stroke_evolution, cavity_motion = Thermodynamics.adiabatic_stroke_2(
        state.ρ, [state.c₁, state.c₂], time, Δt, jumps;
        sampling_steps=ss, verbose=1)

    dims = Int(sqrt(size(state.ρ)[1]))
    ρ₁_evolution = [partial_trace(real(ρ), (dims, dims), 1) for ρ in stroke_evolution]
    ρ₂_evolution = [partial_trace(real(ρ), (dims, dims), 2) for ρ in stroke_evolution]
    c₁_lengths = [l1 for (l1, _) in cavity_motion]
    c₂_lengths = [l2 for (_, l2) in cavity_motion]
    
    append!(state.ρ₁_evolution, ρ₁_evolution)
    append!(state.ρ₂_evolution, ρ₂_evolution)
    append!(state.c₁_evolution, c₁_lengths)
    append!(state.c₂_evolution, c₂_lengths)
    
    state.ρ = real(chop!(stroke_evolution[end]))
    state.c₁.length = cavity_motion[end][1]
    state.c₂.length = cavity_motion[end][2]
    return state, stroke_evolution
end


function cycle(state, 
        isochore_t, isochore_samplings, heat_params, cool_params, 
        adiabatic_t, adiabatic_samplings, adiabatic_params)
    
    # Isochoric Heating
    ga_h, gb_h, bosonic_h  = heat_params
    state, _ = _phaseonium_stroke(state, isochore_t, bosonic_h, ga_h, gb_h, isochore_samplings)
    # Adiabatic Expansion
    Δt, a, ad = adiabatic_params
    state, _ = _adiabatic_stroke(state, adiabatic_t, Δt, [a, ad], adiabatic_samplings)
    # Isochoric Cooling
    ga_c, gb_c, bosonic_c = cool_params
    state, _ = _phaseonium_stroke(state, isochore_t, bosonic_c, ga_c, gb_c, isochore_samplings)
    # Adiabatic Compression
    state, _ = _adiabatic_stroke(state, adiabatic_t, Δt, [a, ad], adiabatic_samplings)
    
    return state
end


function last_cycle(state, 
        isochore_t, isochore_samplings, heat_params, cool_params, 
        adiabatic_t, adiabatic_samplings, adiabatic_params)

    strokes_evolution = []
    # Isochoric Heating
    ga_h, gb_h, bosonic_h  = heat_params
    state, stroke_evolution = _phaseonium_stroke(state, isochore_t, bosonic_h, ga_h, gb_h, isochore_samplings)
    push!(strokes_evolution, stroke_evolution)
    # Adiabatic Expansion
    Δt, a, ad = adiabatic_params
    state, stroke_evolution = _adiabatic_stroke(state, adiabatic_t, Δt, [a, ad], adiabatic_samplings)
    push!(strokes_evolution, stroke_evolution)
    # Isochoric Cooling
    ga_c, gb_c, bosonic_c = cool_params
    state, stroke_evolution = _phaseonium_stroke(state, isochore_t, bosonic_c, ga_c, gb_c, isochore_samplings)
    push!(strokes_evolution, stroke_evolution)
    # Adiabatic Compression
    state, stroke_evolution = _adiabatic_stroke(state, adiabatic_t, Δt, [a, ad], adiabatic_samplings)
    push!(strokes_evolution, stroke_evolution)
    
    return state, strokes_evolution
end


function run_cycle(options)
    cavity1 = create_cavity(options.cavity)
    cavity2 = create_cavity(options.cavity)
    ω = options.cavity.alpha / options.cavity.length
    
    # Joint system
    ρ_tot = create_joint_system(options, ω)
    # Jump Operators
    a = BosonicOperators.destroy(options.dims)
    ad = BosonicOperators.create(options.dims)
    
    # Phaseonium atoms
    heat_params, cool_params = create_phaseoniums(options, ω)

    # State object comprising the cavities parameters and density matrices, 
    # as well as their temporal evolution
    state, past_cycles = load_or_create_state(options.loading, ρ_tot, cavity1, cavity2)

    for t in 1:options.cycles
        println("Cycle $(t + past_cycles)")
        if t == options.cycles
            state = cycle(
                state, 
                options.stroke_time.isochore, options.samplings.isochore, heat_params, cool_params,
                options.stroke_time.adiabatic, options.samplings.adiabatic, (options.dt, a, ad))
        else
            state, evolution = cycle(
                state, 
                options.stroke_time.isochore, options.samplings.isochore, heat_params, cool_params,
                options.stroke_time.adiabatic, options.samplings.adiabatic, (options.dt, a, ad))

            # Save evolution of composite system in the last cycle
            serialize("$(ARGS[1])/system_evolution", evolution)
        end
    end

    # Save state
    serialize("$(ARGS[1])/state_cascade_nonthermal_$(options.cycles + past_cycles)cycles", state)
end

function main()
    run_cycle(read_configuration(ARGS[1]))    
end

main()
