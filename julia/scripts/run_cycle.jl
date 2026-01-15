using Serialization: serialize_array_data
"""
Suggested workflow:

Created in init.jl and globally available:
`config`::OneCavConfig = dict-like struct with configuration options from config.toml
`evolution`::StrokeState = struct created in initialization with initial thermal ρ and cavity;
`cavity`::Cavity = cavity struct containing all the cavity settings

Run the evolution of the cavity state step by step throughout the cycle:
## Cycle 0-1
`ρ`, `ρ_evolution` = `thermalize_by_phaseonium("heating", cavity, config, ρ0=evolution.ρ)`
### Options
 - use `load=true` to load the thermalized state from the path "data/stepbystep_evolution/state_1_thermalized.jl"
 - use `verbose=true` to plot the temperature evolution and save it in "img" folder.

## Save evolution
This will save a StrokeState with all the history of the evolution 
and the final states of the cavity and the field.
`cavity_evolution` = `[cavity.length for _ in 1:length(ρ_evolution)]`
`update_evolution!(evolution, ρ_evolution, cavity_evolution)`
`save_evolution(evolution)`
Run the following block to execute the whole workflow
`
begin
r, revo = thermalize_by_phaseonium("heating", cavity, config, ρ0=evolution.ρ)
cevo = [cavity.length for _ in 1:length(revo)]
time = collect(range(config.Δt, config.Δt*config.time.isochore, length=config.samplings.isochore))
update_evolution!(evolution, revo, cevo, time)
save_evolution(evolution, 1)
end
`

---
# Cycle 1-2
Run the following block to evolve the cavity and save the evolution
`
begin
time, cevo = move_by_pressure("expansion", cavity,config,ρ=evolution.ρ,verbose=true);
revo = [evolution.ρ for _ in 1:length(cevo)]
time=[evolution.time[end]+t for t in time]
update_evolution!(evolution, revo, cevo, time)
save_evolution(evolution, 2)
end
`

---
# Cycle 2-3
Cool down the cavity with phaseonium atoms.
Run the following block to evolve the cavity and save the evolution
`
begin
revo, cevo, time = thermalize_by_phaseonium("cooling",cavity,config,ρ0=evolution.ρ,verbose=true);
update_evolution!(evolution, revo, cevo, time)
save_evolution(evolution, 2)
end
`
"""


include("./init.jl")

function thermalize_by_phaseonium(process, cavity, config; ρ0=nothing, load=false, verbose=false)
  if load
    if process == "heating"
      step = 1
    elseif process == "cooling"
      step = 3
    end
    evolution = deserialize("data/stepbystep_evolution/evolution_$(step-1)$(step).jl")
    ρ = evolution.ρ
  else
    ω0 = cavity.α / cavity.length
    if isnothing(ρ0)
      ρ0 = thermalstate(config.dims, ω0, config.T_initial)
    end

    if process == "heating"
      ϕ = config.phaseonium.ϕ_h
      α = alpha_from_temperature(config.phaseonium.T_hot, ϕ, ω0)
    elseif process == "cooling"
      ϕ = config.phaseonium.ϕ_c
      α = alpha_from_temperature(config.phaseonium.T_cold, ϕ, ω0)
    end
    ga, gb = dissipationrates(α, ϕ)
    kraus = kraus_operators(
      ga, gb, config.Ω, config.Δt, config.dims)
    kraus_dag = [k' for k in kraus]

    # Check that the Sparse Arrays are usable in this case
    println("Sparsity in the starting state:")
    println(count(==(0), ρ0) / length(ρ0))
    ρ0 = sparse(ρ0)

    collisions = config.time.isochore
    ρ, evolution, temperatures = thermalization_stroke(
      ρ0, kraus, kraus_dag, collisions, config.samplings.isochore, ω0)

    if verbose
      g = plot(temperatures)
      savefig(g, "img/temperature_stroke_1.png")
    end

    final_t = Phaseonium.Measurements.temperature(ρ, ω0)
    println("Final temperature of the state: $final_t")
  end

  return ρ, evolution
end


function update_evolution!(
  evolution, state_evolution, cavity_evolution;
  initial_state=nothing, initial_cavity=nothing)
  # Add to the stroke state the evolution of the state density matrix
  if isnothing(initial_state)
    initial_state = evolution.ρ
  end
  if isnothing(initial_cavity)
    initial_cavity = evolution.c₁.length
  end
  pushfirst!(state_evolution, initial_state)
  append!(evolution.ρ₁_evolution, state_evolution)
  # Add to the stroke state the evolution of the cavity length
  prepend!(cavity_evolution, initial_cavity)
  append!(evolution.c₁_evolution, cavity_evolution)
  # Update the 'current' state of the system and cavity
  evolution.ρ = state_evolution[end]
  evolution.c₁.length = cavity_evolution[end]

  return evolution
end


function save_evolution(evolution::StrokeState, step)
  open("data/stepbystep_evolution/evolution_$(step-1)$(step).jl", "w") do f
    serialize(f, evolution)
  end
  println("Evolution object saved in data/stepbystep_evolution/evolution_$(step-1)$(step).jl")
end


