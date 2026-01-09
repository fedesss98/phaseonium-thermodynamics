include("./init.jl")

function thermalize_by_phaseonium(cavity, config; ρ0=nothing, load=false)
    if load
        evolution = deserialize("data/stepbystep_evolution/state_1_thermalized.jl")
        ρ = evolution.ρ
    else
        ω0 = cavity.α / cavity.length
        if isnothing(ρ0) 
          ρ0 = thermalstate(config.dims, ω0, config.T_initial)
        end

        ϕ = config.phaseonium.ϕ_h
        α = alpha_from_temperature(config.T_initial, ϕ, ω0)
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
        plot(temperatures)
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
        initial_cavity = evolution.c₁
    end
    prepend!(state_evolution, initial_state)
    append!(evolution.ρ₁_evolution, state_evolution)
    # Add to the stroke state the evolution of the cavity length
    prepend!(cavity_evolution, initial_cavity)
    append!(evolution.c₁_evolution, cavity_evolution)
end
