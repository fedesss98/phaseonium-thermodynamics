using Phaseonium
#using CSV
#using DataFrames
using TOML
using Configurations
using Serialization
using SparseArrays

#using QuantumToolbox

#using ProgressBars
using ProgressMeter

using Plots
using LaTeXStrings

include("../src/RoutineFunctions.jl")

@option struct CavityConfig
  α::Float64
  l_min::Float64
  l_max::Float64
  surface::Float64
  acceleration::Float64
  expanding_force::Float64
  compressing_force::Float64
  friction::Float64 = 0.0
  mass::Float64

  function CavityConfig(config_dict::Dict)
    new(
      config_dict["alpha"],
      config_dict["min_length"],
      config_dict["max_length"],
      config_dict["surface"],
      config_dict["acceleration"],
      config_dict["expanding_force"],
      config_dict["compressing_force"],
      config_dict["friction"],
      config_dict["mass"],
    )
  end
end

@option struct PhaseonimConfig
  ϕ_c::Float64
  T_cold::Float64
  ϕ_h::Float64
  T_hot::Float64
  function PhaseonimConfig(config_dict::Dict)
    new(
      config_dict["phi_cold"],
      config_dict["T_cold"],
      config_dict["phi_hot"],
      config_dict["T_hot"],
    )
  end
end

@option struct StrokeTimeConfig
  isochore::Int
  adiabatic::Int
end


@option struct SamplingsConfig
  isochore::Int
  adiabatic::Int
end


@option struct ReloadConfig
  load_state::Bool
  filename::String
  past_cycles::Int
end

@option struct OneCavConfig
  dims::Int
  Ω::Float64
  Δt::Float64
  T_initial::Float64
  cavity::CavityConfig
  phaseonium::PhaseonimConfig
  time::StrokeTimeConfig
  samplings::SamplingsConfig
  loading::ReloadConfig

  function OneCavConfig(config_dict::Dict)
    new(
      config_dict["meta"]["dims"],
      config_dict["meta"]["omega"],
      config_dict["meta"]["dt"],
      config_dict["meta"]["T1_initial"],
      CavityConfig(config_dict["cavity1"]),
      PhaseonimConfig(config_dict["phaseonium"]),
      config_dict["stroke_time"],
      config_dict["samplings"],
      config_dict["loading"]
    )
  end
end


function read_configuration(config_file="config.toml")
  config = TOML.parsefile(config_file)

  config = OneCavConfig(config)

  return config
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


config = read_configuration()
fast_config = read_configuration("fast_config.toml")
cavity = create_cavity(config.cavity)
evolution = StrokeState(
  thermalstate(config.dims, cavity.α / cavity.length, config.T_initial),
  cavity
)
