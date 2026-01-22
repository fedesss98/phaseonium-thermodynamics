"""
Suggested workflow:

Created in init.jl and globally available:
`config`::OneCavConfig = dict-like struct with configuration options from config.toml
`evolution`::StrokeState = struct created in initialization with initial thermal ρ and cavity;
`cavity`::Cavity = cavity struct containing all the cavity settings

Run the evolution of the cavity state step by step throughout the cycle
(choose the `stroke_function` accordingly).
`
begin
step = 2
process = "expansion"
revo, cevo, time = stroke_function(process,cavity,config,ρ0=evolution.ρ,verbose=true);
update_evolution!(evolution, revo, cevo, time)
save_evolution(evolution, step)
end
`

All the cycle can be automated with the `cycle` function.

`
begin
n_cycles=5
evolution = cycle(config, n_cycles, cavity=evolution.c₁, ρ0=evolution.ρ, verbose=true)
println("\nSIMULATION COMPLETED")
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
    ρ0 = Matrix(thermalstate(config.dims, ω0, config.T_initial))
  end

  α = alpha_from_temperature(temp, ϕ, ω0)
  ga, gb = dissipationrates(α, ϕ)
  kraus = kraus_operators(
    ga, gb, config.Ω, config.Δt, config.dims)
  kraus_dag = [k' for k in kraus]

  if verbose
    println("\n--- $step) Isochoric Thermalization ---")
    println("Initial Temperature: $(Phaseonium.Measurements.temperature(ρ0, ω0))")
    println("Target Temperature: $temp")
    # Check that the Sparse Arrays are usable in this case
    println("Sparsity in the starting state:")
    println(count(==(0), ρ0) / length(ρ0))
  end
  #ρ0 = sparse(ρ0)

  collisions = config.time.isochore
  ρ, evolution, temperatures = thermalization_stroke(
    ρ0, kraus, kraus_dag, collisions, config.samplings.isochore, ω0)

  if verbose
    g = plot(temperatures)
    savefig(g, "img/temperature_stroke_$step.png")
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
    println("\n--- $step) Adiabatic $(uppercasefirst(process)) ---")
    println("Initial Length: $l0")
    println("Target Max Length: $target_l")
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
    println("Final Length: $(cavity_evolution[end])")
    println("Final time: $(time_steps[end])")

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


function save_evolution(evolution::StrokeState, step; save_in=nothing)
  if isnothing(save_in)
    fname = "data/evolution_step$(step-1)$(step).jl"
  else
    fname = "data/$save_in/evolution_step$(step-1)$(step).jl"
  end
  open(fname, "w") do f
    serialize(f, evolution)
  end
  println("Evolution object saved in $fname")
end


function load_evolution(step; load_from=nothing)
  if isnothing(load_from)
    fname = "data/evolution_step$(step-1)$(step).jl"
  else
    fname = "data/$load_from/evolution_step$(step-1)$(step).jl"
  end
  evolution = deserialize(fname)
  return evolution
end


"""
  reset_evolution(evolution::StrokeState)

Reset the evolution of the system keeping the only the starting state and the cavity.
"""
function reset_evolution!(evolution::StrokeState)
  T = promote_type(eltype(evolution.ρ), ComplexF64)
  evolution.ρ₁_evolution = Vector{Matrix{T}}()
  evolution.c₁_evolution = Float64[]
  evolution.time = Float64[]
end


function plot_evolution(temperatures, entropies, times; save_in=nothing, name="cycle", title="")

  p1 = plot(times, ylabel="Evolution Time")
  p2 = plot(entropies, temperatures, xlabel="Entropy", ylabel="Temperature")
  # Add starting and ending points of the cycle
  scatter!(p2, [entropies[1]], [temperatures[1]],
    color=:green,
    markershape=:circle,
    markersize=6,
    label="Start")
  scatter!(p2, [entropies[end]], [temperatures[end]],
    color=:red,
    markershape=:rect,
    markersize=6,
    label="End")
  p3 = plot(times, temperatures, xlabel="Time", ylabel="Temperature")
  p4 = plot(times, entropies, xlabel="Time", ylabel="Entropy")
  p = plot(
    p1, p2, p3, p4,
    layout=(4, 1), size=(600, 900),
    plot_title=title
  )
  if !isnothing(save_in)
    savefig(p, "img/$save_in/$name.png")
  end
end


function plot_evolution(evolution; kwargs...)
  α0 = evolution.c₁.α

  temperatures = Phaseonium.Measurements.temperature.(
    evolution.ρ₁_evolution, α0 ./ evolution.c₁_evolution)
  entropies = Phaseonium.Measurements.entropy_vn.(
    evolution.ρ₁_evolution)

  plot_evolution(temperatures, entropies, evolution.time; kwargs...)
end


function cycle(config, n_cycles=1; cavity=nothing, ρ0=nothing, verbose=false, reload_from_step=0)

  experiment = config.name

  current_rho = nothing
  current_time = 0.0
  current_cavity = nothing
  temperatures = Float64[]
  entropies = Float64[]
  times = Float64[]

  if reload_from_step > 0
    # Load one saved state to resume calculation
    evolution = load_evolution(reload_from_step, load_from=experiment)
    # Extract only final states
    current_rho = evolution.ρ
    current_time = evolution.time[end]
    current_cavity = evolution.c₁

    # Clean up memory
    evolution = nothing
    GC.gc()
  else
    if isnothing(ρ0)
      ρ0 = thermalstate(config.dims, cavity.α / cavity.length, config.T_initial)
    end
    if isnothing(cavity)
      cavity = create_cavity(config.cavity)
    end

    current_rho = ρ0
    current_time = 0.0
    current_cavity = cavity

  end

  # Utility function to correctly route strokes functions
  function _route_process(process, state, cavity, config, verbose)
    if process in ["heating", "cooling"]
      revo, cevo, time = thermalize_by_phaseonium(process, cavity, config, ρ0=state, verbose=verbose)
    elseif process in ["expansion", "compression"]
      revo, cevo, time = move_by_pressure(process, cavity, config, ρ0=state, verbose=verbose)
    end
    return revo, cevo, time
  end

  # === Main loop ===

  processes = ["heating", "expansion", "cooling", "compression"]

  total_steps = 1  # Total number of steps, considering cycles
  for cycle_i in 1:n_cycles
    println("\n=== CYCLE $cycle_i ===\n")

    for (step, process) in enumerate(processes)
      if total_steps <= reload_from_step
        continue
      end

      s_evolution, c_evolution, time = _route_process(process, current_rho, current_cavity, config, verbose)
      cavity.length = c_evolution[end]
      time = [current_time + t for t in time]

      step_evolution = StrokeState(
        s_evolution[end],
        cavity
      )
      append!(step_evolution.ρ₁_evolution, s_evolution)
      append!(step_evolution.c₁_evolution, c_evolution)
      append!(step_evolution.time, time)
      save_evolution(step_evolution, total_steps, save_in=experiment)
      if verbose
        append!(temperatures, [
          Phaseonium.Measurements.temperature(rho, cavity.α / l)
          for (rho, l) in zip(s_evolution, c_evolution)
        ])
        append!(entropies, [
          Phaseonium.Measurements.entropy_vn(Matrix(rho))
          for rho in s_evolution
        ])
        append!(times, time)
      end
      total_steps += 1

      current_rho = s_evolution[end]
      current_cavity = cavity
      current_time = time[end]
    end
  end

  if verbose
    plot_evolution(temperatures, entropies, times, save_in=experiment)
  end

  return current_rho, current_cavity, current_time, total_steps

end


function plot_saved_evolution(config; returns=false, from_cycle=1)
  temperatures = Float64[]
  entropies = Float64[]
  lengths = Float64[]
  times = Float64[]
  α = cavity.α
  for step in 1:4
    evo = load_evolution(4 * (from_cycle - 1) + step, load_from=config.name)
    # Skip the first element as it is equal to the last in previous step
    step_temperatures = [
      temperature(r, α / l) for (r, l) in zip(evo.ρ₁_evolution[2:end], evo.c₁_evolution[2:end])]
    append!(temperatures, step_temperatures)
    step_entropies = [entropy_vn(Matrix(r)) for r in evo.ρ₁_evolution[2:end]]
    append!(entropies, step_entropies)
    append!(lengths, [l for l in evo.c₁_evolution[2:end]])
    append!(times, [t for t in evo.time])
  end
  plot_evolution(temperatures, entropies, times, save_in=config.name, title="T-S evolution from cycle $from_cycle")

  if returns
    return temperatures, lengths, entropies, times
  end
end


# Warmup the thermalization loop
println("Warming up...")
_ = thermalize_by_phaseonium("heating", cavity, fast_config);
