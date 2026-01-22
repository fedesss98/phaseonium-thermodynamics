using Phaseonium
#using CSV
#using DataFrames
using TOML
using Configurations
using Accessors  # Used to create new configs: new_config = @set config.cavity.α = 2.5
using Serialization
using SparseArrays

#using QuantumToolbox

#using ProgressBars
using ProgressMeter

using Plots
using LaTeXStrings

include("../src/RoutineFunctions.jl")


Base.@kwdef struct CavityConfig
  α::Float64
  l_min::Float64
  l_max::Float64
  surface::Float64
  acceleration::Float64
  expanding_force::Float64
  compressing_force::Float64
  friction::Float64 = 0.0
  mass::Float64
end

Base.@kwdef struct PhaseoniumConfig
  ϕ_c::Float64
  T_cold::Float64
  ϕ_h::Float64
  T_hot::Float64
end

Base.@kwdef struct StrokeTimeConfig
  isochore::Int
  adiabatic::Int
end

Base.@kwdef struct SamplingsConfig
  isochore::Int
  adiabatic::Int
end

Base.@kwdef struct ReloadConfig
  load_state::Bool
  filename::String = ""
  past_cycles::Int = 0
end

# --- Main Struct ---

Base.@kwdef struct OneCavConfig
  # Fields flattened from [meta]
  name::String
  description::String
  dims::Int
  Ω::Float64
  Δt::Float64
  T_initial::Float64

  # Nested structs
  cavity::CavityConfig
  phaseonium::PhaseoniumConfig
  time::StrokeTimeConfig
  samplings::SamplingsConfig
  loading::ReloadConfig
end

function read_configuration(config_file="config.toml")
  # Parse raw TOML
  d = TOML.parsefile(config_file)

  meta = d["meta"]
  cav = d["cavity1"]
  ph = d["phaseonium"]
  st = d["stroke_time"]
  sm = d["samplings"]
  ld = d["loading"]

  cavity_config = CavityConfig(
    α=cav["alpha"],
    l_min=cav["min_length"],
    l_max=cav["max_length"],
    surface=cav["surface"],
    acceleration=cav["acceleration"],
    expanding_force=cav["expanding_force"],
    compressing_force=cav["compressing_force"],
    friction=get(cav, "friction", 0.0),
    mass=cav["mass"]
  )

  phaseonium_config = PhaseoniumConfig(
    ϕ_c=ph["phi_cold"],
    T_cold=ph["T_cold"],
    ϕ_h=ph["phi_hot"],
    T_hot=ph["T_hot"]
  )

  stroke_time_config = StrokeTimeConfig(
    isochore=st["isochore"],
    adiabatic=st["adiabatic"]
  )

  samplings_config = SamplingsConfig(
    isochore=sm["isochore"],
    adiabatic=sm["adiabatic"]
  )

  reload_config = ReloadConfig(
    load_state=ld["load_state"],
    filename=get(ld, "filename", ""),
    past_cycles=get(ld, "past_cycles", 0)
  )

  return OneCavConfig(
    name=get(meta, "name", ""),
    description=get(meta, "description", ""),
    dims=meta["dims"],
    Ω=meta["omega"],
    Δt=meta["dt"],
    T_initial=meta["T1_initial"], cavity=cavity_config,
    phaseonium=phaseonium_config,
    time=stroke_time_config,
    samplings=samplings_config,
    loading=reload_config
  )
end


function create_cavity(config)
  return Cavity(config.mass, config.surface,
    config.l_min, config.l_max,
    config.α, config.friction,
    config.expanding_force, config.compressing_force)
end


function get_omega(cavity)
  return cavity.α / cavity.length
end

println(ARGS[1])
!isempty(ARGS) ? config_file = ARGS[1] : config_file = "config.toml"
config = read_configuration(config_file)
experiment = config.name
mkpath("data/$experiment")
mkpath("img/$experiment")
println("Experiment $experiment initialized.")
fast_config = read_configuration("fast_config.toml")
cavity = create_cavity(config.cavity)
ρ0 = thermalstate(config.dims, cavity.α / cavity.length, config.T_initial)
evolution = StrokeState(
  ρ0,
  cavity
)
time = 0.0
append!(evolution.time, time)

