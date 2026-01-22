module Strokes

using DifferentialEquations

export StrokeState, thermalization_stroke, adiabatic_stroke

using ..OpticalCavity
using ..Measurements
using ProgressMeter
using SparseArrays
using LinearAlgebra


mutable struct StrokeState{T<:Complex}
    ρ::Matrix{T}
    c₁::Cavity
    c₂::Union{Cavity, Nothing}
    ρ₁_evolution::Vector{Matrix{T}}
    ρ₂_evolution::Vector{Matrix{T}}
    c₁_evolution::Vector{Float64}
    c₂_evolution::Vector{Float64}
    time::Vector{Float64}

    # Two-cavity constructor
    function StrokeState(ρ::Matrix{T}, c1::Cavity, c2::Cavity) where {T<:Complex}
        new{T}(
            ρ, c1, c2,
            Vector{Matrix{T}}(), Vector{Matrix{T}}(),
            Float64[], Float64[], Float64[]
        )
    end

    # One-cavity constructor
    function StrokeState(ρ::AbstractMatrix, c1::Cavity)
        T = promote_type(eltype(ρ), ComplexF64)
        ρ_dense = Matrix{T}(ρ)
        new{T}(
            ρ_dense, c1, nothing,
            Vector{Matrix{T}}(), Vector{Matrix{T}}(),
            Float64[], Float64[], Float64[]
        )
    end
end


"""
===== CYCLE EVOLUTION =====
"""

"""
    thermalization_stroke(ρ, kraus, kraus_dag, collisions, sampling_time, ω)

Evolve the state of a cavity field using the Phaseonium collisional thermal map.

- `kraus`: Kraus operators C, C', S and S dagger for the map
- `kraus_dag`: Complex Conjugate of Kraus operators
- `collisions`: Total number of collisions for the evolution
- `n_samplings`: Number of times in which to save the evolution status
- `ω`: Frequency of the cavity, used to sample the temperature
"""
function thermalization_stroke(ρ, kraus, kraus_dag, collisions, n_samplings, ω)
  # Pre-allocate temperature array
  sampling_times = unique(round.(Int, range(collisions/n_samplings, collisions, length=n_samplings)))
  n_actual = length(sampling_times)  # Recalculate in case 'unique' removed duplicates

  temperatures = Vector{Float64}(undef, n_actual)
  evolution = Vector{typeof(ρ)}(undef, n_actual)
  save_cursor = 1

  ρ_next = similar(ρ)
  buffer = similar(ρ)
  
  @showprogress for k in 1:collisions
      
    # --- KRAUS MAP: ρ_new = Σ E_i * ρ * E_i' ---
    # Reset the accumulator with the first term of the sum
    mul!(buffer, kraus[1], ρ)
    mul!(ρ_next, buffer, kraus_dag[1])
    
    # Add remaining terms
    for i in 2:length(kraus)
      mul!(buffer, kraus[i], ρ)
      # The 5-args mul!(C, A, B, a, b) computes C = aAB + bC
      mul!(ρ_next, buffer, kraus_dag[i], 1, 1)
    end
    
    # Swap pointers
    ρ, ρ_next = ρ_next, ρ

    if save_cursor <= n_actual && k == sampling_times[save_cursor]
      temperatures[save_cursor] = Measurements.temperature(ρ, ω)
      evolution[save_cursor] = copy(ρ)
      # Clean up tiny numerical noise that might ruin sparsity
      #dropzeros!(ρ) 
      # Advance the cursor and wait for next target
      save_cursor += 1
    end
  end
  
  return ρ, evolution, temperatures
end


"""
  adiabatic_stroke(avg_n; l0, v0, target_l, cavity, external_force, n_samplings, max_time=1e4)

Evolve the cavity length under the radiation pressure force and external force.
The evolution is solved as an Ordinary Differential Equation, supposing the state remains constant during the adiabatic stroke 
(no dissipation, constant number of photons)
# Args
 - `avg_n`: expectation value of the Number Operator for the current state of the field
 - `l0`: initial length of the cavity
 - `v0`: initial velocity of the cavity (usually starting still)
 - `target_l`: final length of the cavity
 - `cavity`: cavity object with all its parameters
 - `external_force`: external force on the cavity wall
 - `n_samplings`: number of times in which to sample the evolution of the cavity
 - `max_time`: maximum time for the computation of the solution of the ODE. Must be greater than the time needed to reach the target lenght
"""
function adiabatic_stroke(
  avg_n; 
  l0, v0, target_l, cavity, external_force, n_samplings, max_time=1e4)

  function piston_dynamics!(du, u, args, t)
      L = u[1]
      v = u[2]
      avg_n = args[1]
      F_ext = args[2]
      α = args[3]
      m = args[4]
      γ_damping = args[5]

      # Radiation Force Definition
      # F = -dE/dL = (ħ * α / L^2) * (<n> + 1/2)
      # The +0.5 is the vacuum energy contribution explicitly kept in Tejero [cite: 68]
      F_rad = (α0 / L^2) * (avg_n + 0.5)
      
      # Net Force
      F_net = F_rad - F_ext - γ_damping * v

      du[1] = v
      du[2] = F_net / m
  end

  # B. The Termination Condition (Callback)
  # This function triggers when it returns 0. 
  # We want it to trigger when L(t) - L_target = 0.
  condition(u, t, integrator) = u[1] - target_l

  # What to do when triggered: Stop the integrator
  affect!(integrator) = terminate!(integrator)

  # Create the callback
  # "continuous" means the solver will interpolate to find the EXACT time L hits L_max
  cb = ContinuousCallback(condition, affect!)

  # Initial state vector for ODE [L, v]
  u0 = [l0, v0]

  # Time span (Make it large enough to ensure we reach L_max, the callback will stop it early)
  t_span = (0.0, max_time)

  α0 = cavity.α
  surface = cavity.surface
  args = [avg_n, external_force, α0, cavity.mass, cavity.γ]
  # Define the problem
  prob = ODEProblem(piston_dynamics!, u0, t_span, args)

  # Solve with the callback
  sol = solve(prob, Tsit5(), callback=cb, reltol=1e-8, abstol=1e-8)

  # Reconstruct the evolution of the cavity
  # Do not save initial values at t0
  t_grid = range(sol.t[2], sol.t[end], length=n_samplings)
  sampled_solutions = sol(t_grid)
  cavity_evolution = [u[1] for u in sampled_solutions]
  time_steps = collect(t_grid)

  return time_steps, cavity_evolution

end


function _phaseonium_stroke(state::StrokeState, ndims, time, bosonic, ga, gb, samplingssteps, io)
    if state.c₂ === nothing
        # Single system evolution
        stroke_evolution = Thermodynamics.phaseonium_stroke(
            state.ρ, time, bosonic, [ga, gb]; sampling_steps=50, verbose=1)

        c₁_lengths = [state.c₁.length for _ in stroke_evolution]

        append!(state.ρ₁_evolution, stroke_evolution)
        append!(state.c₁_evolution, c₁_lengths)
    else
        # Two cavities evolution
        stroke_evolution = Thermodynamics.phaseonium_stroke_2(
            state.ρ, time, bosonic, ga, gb; 
            sampling_steps=samplingssteps, verbose=1, io=io)
    
        ρ₁_evolution = [partial_trace(real(ρ), (ndims, ndims), 1) for ρ in stroke_evolution]
        ρ₂_evolution = [partial_trace(real(ρ), (ndims, ndims), 2) for ρ in stroke_evolution]
        c₁_lengths = [state.c₁.length for _ in stroke_evolution]
        c₂_lengths = [state.c₂.length for _ in stroke_evolution]
        
        append!(state.ρ₁_evolution, ρ₁_evolution)
        append!(state.ρ₂_evolution, ρ₂_evolution)
        append!(state.c₁_evolution, c₁_lengths)
        append!(state.c₂_evolution, c₂_lengths)
    end

    # state.ρ = real(chop!(stroke_evolution[end]))
    state.ρ = copy(stroke_evolution[end])
    # Jump Operators
    # n = BosonicOperators.create(ndims) * BosonicOperators.destroy(ndims)
    # Print number of photons
    # println("Average Photons: $(tr(state.ρ * kron(n, n)))")

    return state, stroke_evolution
end


function _adiabatic_stroke(state::StrokeState, jumps, ndims, Δt, samplingssteps, process, io)
    if state.c₂ === nothing
        println("Solving one cavity adiabatic ODE...")
        # We suppose the number of photons is kept constant throughout the adiabatic process
        n = jumps[2] * jumps[1]  # ad*a
        avg_n = real(tr(n * state.ρ))
        stroke_evolution, 
        cavity_motion, 
        cavity_velocities,
        total_time = Thermodynamics.adiabatic_stroke_ode(
            state.ρ, avg_n, state.c₁;
            sampling_steps=samplingssteps, max_time=1e6, verbose=2, io=io)
            
        append!(state.ρ₁_evolution, stroke_evolution)
        append!(state.c₁_evolution, cavity_motion)
        
        state.c₁.length = cavity_motion[end]
    else
        stroke_evolution, 
        cavity_motion, 
        total_time = Thermodynamics.adiabatic_stroke_2(
            state.ρ, (state.c₁, state.c₂), jumps, Δt, process;
            sampling_steps=samplingssteps, verbose=3, io=io)
    
        ρ₁_evolution = [partial_trace(real(ρ), (ndims, ndims), 1) for ρ in stroke_evolution]
        ρ₂_evolution = [partial_trace(real(ρ), (ndims, ndims), 2) for ρ in stroke_evolution]
        c₁_lengths = [l1 for (l1, _) in cavity_motion]
        c₂_lengths = [l2 for (_, l2) in cavity_motion]
        
        append!(state.ρ₁_evolution, ρ₁_evolution)
        append!(state.ρ₂_evolution, ρ₂_evolution)
        append!(state.c₁_evolution, c₁_lengths)
        append!(state.c₂_evolution, c₂_lengths)
        
        state.c₁.length = cavity_motion[end][1]
        state.c₂.length = cavity_motion[end][2]
    end
    
    # state.ρ = real(chop!(stroke_evolution[end]))
    state.ρ = copy(stroke_evolution[end])
    return state, stroke_evolution, total_time
end



function cycle(state, Δt, system_evolutions, cycle_steps, isochore_t, isochore_samplings, adiabatic_t, adiabatic_samplings, io)
    if state isa Vector
        ρ, c₁, c₂ = state
        state = StrokeState(Matrix(ρ), c₁, c₂)
    end
        
    if state.c₂ === nothing
        ndims = Int64(size(state.ρ)[1])
    else
        ndims = Int64(sqrt(size(state.ρ)[1]))  # Dimensions of one cavity
    end
    # Jump Operators
    a = BosonicOperators.destroy(ndims)
    ad = BosonicOperators.create(ndims)
    jumps = (a, ad)
    
    # Isochoric Heating
    state, system_evolution = _phaseonium_stroke(state, ndims, isochore_t, bosonic_h, ga_h, gb_h, isochore_samplings, io)
    append!(system_evolutions, system_evolution)
    append!(cycle_steps, Δt*isochore_t)
    # Adiabatic Expansion
    state, system_evolution, adiabatic_t = _adiabatic_stroke(state, jumps, ndims, Δt, adiabatic_samplings, "Expansion", io)
    append!(system_evolutions, system_evolution)
    append!(cycle_steps, cycle_steps[end] + adiabatic_t)
    # Isochoric Cooling
    state, system_evolution = _phaseonium_stroke(state, ndims, isochore_t, bosonic_c, ga_c, gb_c, isochore_samplings, io)
    append!(system_evolutions, system_evolution)
    append!(cycle_steps, cycle_steps[end] + Δt*isochore_t)
    # Adiabatic Compression
    state, system_evolution, adiabatic_t = _adiabatic_stroke(state, jumps, ndims, Δt, adiabatic_samplings, "Compression", io)
    append!(system_evolutions, system_evolution)
    append!(cycle_steps, cycle_steps[end] + adiabatic_t)
    
    return state, system_evolutions
end

end
