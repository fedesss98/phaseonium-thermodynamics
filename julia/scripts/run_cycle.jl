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


function thermalize_by_phaseonium(process, cavity, config; ρ0=nothing, verbose=false)
  # Initialize variables for current step
  if process == "heating"
    step = 1
    ϕ = config.phaseonium.ϕ_h
    temp = config.phaseonium.T_hot
  elseif process == "cooling"
    step = 3
    ϕ = config.phaseonium.ϕ_c
    temp = config.phaseonium.T_cold
  end

  ω0 = cavity.α / cavity.length
  if isnothing(ρ0)
    ρ0 = thermalstate(config.dims, ω0, config.T_initial)
  end

  α = alpha_from_temperature(temp, ϕ, ω0)
  ga, gb = dissipationrates(α, ϕ)
  kraus = kraus_operators(
    ga, gb, config.Ω, config.Δt, config.dims)
  kraus_dag = [k' for k in kraus]

  if verbose
    println("--- $step) Isochoric Thermalization ---")
    println("Initial Temperature: $(Phaseonium.Measurements.temperature(ρ0, ω0))")
    println("Target Temperature: $temp")
    # Check that the Sparse Arrays are usable in this case
    println("Sparsity in the starting state:")
    println(count(==(0), ρ0) / length(ρ0))
  end
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

  cavity_evolution = [cavity.length for _ in 1:length(evolution)]
  time = collect(range(config.Δt, config.Δt * collisions, length=config.samplings.isochore))
  return evolution, cavity_evolution, time
end


function move_by_pressure(process, cavity, config; ρ0=nothing, verbose=false)
  # Set external force and final length based on the process
  if process == "expansion"
    step = 2
    external_force = cavity.expanding_force
    target_l = cavity.l_max
  elseif process == "compression"
    step = 4
    external_force = cavity.compressing_force
    target_l = cavity.l_min
  end

  ω = cavity.α / cavity.length
  if isnothing(ρ0)
    ρ0 = thermalstate(config.dims, ω, config.T_initial)
  end

  l0 = cavity.length  # Starting length
  # The average number of photons is conserved in an adiabatic process,
  # and this number affects the radiation pressure
  avg_n = real(Phaseonium.Measurements.avg_number(ρ0, ω))

  if verbose
    println("--- $step) Adiabatic $(uppercasefirst(process)) ---")
    println("Initial Length: $l0")
    println("Target Max Length: $target_l")
    println("Conserved <n>: $avg_n")
  end

  time_steps, cavity_evolution = adiabatic_stroke(
    avg_n,
    l0=l0,
    v0=0.0,
    target_l=target_l,
    cavity=cavity,
    external_force=external_force,
    n_samplings=config.samplings.adiabatic,
    max_time=1e13
  )

  if verbose
    println("Simulation finished. Final time: $(time_steps[end])")
    println("Final Length: $(cavity_evolution[end])")

    p_evolution = [(cavity.α * (avg_n + 0.5)) / (cavity.surface * l^2) for l in cavity_evolution]
    g1 = plot(time_steps, cavity_evolution, label="Length", ylabel="Length", xlabel="Time")
    g2 = plot(time_steps, p_evolution, label="Pressure", ylabel="Pressure", xlabel="Time")
    p = plot(g1, g2, layout=(2, 1), size=(600, 800))
    savefig(p, "img/length_stroke_$step.png")
  end

  state_evolution = [ρ0 for _ in 1:length(time_steps)]
  return state_evolution, cavity_evolution, time_steps
end


function update_evolution!(
  evolution, state_evolution, cavity_evolution, time;
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
  # Append the time evolution
  append!(evolution.time, time)
  # Update the 'current' state of the system and cavity
  evolution.ρ = state_evolution[end]
  evolution.c₁.length = cavity_evolution[end]

  return
end


function save_evolution(evolution::StrokeState, step, cycle=1)
  fname = "data/stepbystep_evolution/evolution_cycle$(cycle)_step$(step-1)$(step).jl"
  open(fname, "w") do f
    serialize(f, evolution)
  end
  println("Evolution object saved in $fname")
end


