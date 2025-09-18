"""
Try a phaseonium stroke
"""

using Revise

includet("./modules/modules_interface.jl")
includet("./src/RoutineFunctions.jl")
includet("./modules/MasterEquations.jl")


function _check(ρ)
    println("System after the stroke:")
    println(checkdensity(ρ))
    println("Final Temperature of the System: $(Measurements.temperature(ρ, ω))")
end

function plot_stroke_state(system_evolution, cavity_evolution, α0)
    temperatures = []
    entropies = []
    for (i, ρ) in enumerate(system_evolution)
        local ω = α0 / cavity_evolution[i]
        t = Measurements.temperature(ρ, ω)
        ent = Measurements.entropy_vn(ρ)
        
        push!(temperatures, t)
        push!(entropies, ent)
    end

    g = plot(entropies, temperatures, label="Stroke")
    # Plot starting point
    scatter!(g, [entropies[1]], [temperatures[1]], label="Start", mc="blue", ms=5)
    # Plot ending point
    scatter!(g, [entropies[end]], [temperatures[end]], label="End", mc="red", ms=5)
    title!("Phaseonium Stroke (Isochoric)")
    xlabel!("Entropy")
    ylabel!("Temperature")
    display(g)
end


function plot_temperature(system_evolution, cavity_evolution, α0)    
    temperatures = []
    for (i, ρ) in enumerate(system_evolution)
        local ω = α0 / cavity_evolution[i]
        t = Measurements.temperature(ρ, ω)
        
        push!(temperatures, t)
    end

    g = plot(temperatures, label="Temperature")
    title!("Temperature Evolution")
    xlabel!("Time")
    ylabel!("Temperature")

    return g
end


function cascade_evolution(thermalizationtime, ρt, α0, bosonic_operators, ga, gb)
    # system_evolution, cavity_evolution = Thermodynamics.adiabatic_stroke(
    #     ρt, thermalizationtime, Δt, [a, ad], cavity; sampling_each=10)
    system_evolution = Thermodynamics.phaseonium_stroke_2(
        ρt, thermalizationtime, bosonic_operators, ga, gb; sampling_steps=50, verbose=3)

    return system_evolution
end


function onesystem_evolution(thermalizationtime, ρt, α0, bosonic_operators, ga, gb)
    # system_evolution, cavity_evolution = Thermodynamics.adiabatic_stroke(
    #     ρt, thermalizationtime, Δt, [a, ad], cavity; sampling_each=10)
    system_evolution = Thermodynamics.phaseonium_stroke(
        ρt, thermalizationtime, bosonic_operators, [ga, gb]; sampling_steps=50, verbose=3)

    return system_evolution
end


function main(systems, thermalizationtime)
    ndims = 25
    Ω = 1.0
    Δt = 1e-2

    T_initial = 2.0
    T_final = 2.5

    # Create a Cavity
    α0 = 2*π
    l0 = 1.0
    cavity = Cavity(1.0, 1.0, l0, α0, 0, 0.05)
    ω = α0 / l0

    global ρt = thermalstate(ndims, ω, T_initial)

    # Create Phaseonium atoms
    ϕ = π/1.1
    α = Phaseonium.alpha_from_temperature(T_final, ϕ, ω)
    println("Phaseonium α: $α < $(sqrt((1+cos(ϕ))/(3+cos(ϕ))))")

    η = Phaseonium.densitymatrix(α, ϕ)

    ga, gb = Phaseonium.dissipationrates(α, ϕ)
    final_temperature = Phaseonium.finaltemperature(ω, ga, gb)
    println(
        "Apparent Temperature carried by Phaseonium atoms: 
        $(final_temperature)")

    # Define Kraus Operators
    identity_op = I(ndims)
    a = BosonicOperators.destroy(ndims)
    ad = BosonicOperators.create(ndims)

    C = BosonicOperators.C(Ω*Δt, ndims)
    Cp = BosonicOperators.Cp(Ω*Δt, ndims)
    S = BosonicOperators.S(Ω*Δt, ndims)
    Sd = BosonicOperators.Sd(Ω*Δt, ndims)

    E0 = sqrt(1 - ga/2 - gb/2) * identity(ndims)
    E1 = sqrt(ga/2) * C
    E2 = sqrt(ga) * S
    E3 = sqrt(gb/2) * Cp
    E4 = sqrt(gb) * Sd

    kraus = [E0, E1, E2, E3, E4]

    if systems == 1
        println("One system evolution")
        system_evolution = onesystem_evolution(thermalizationtime, ρt, α0, [C, Cp, S, Sd], ga, gb)
        figname = "phaseonium_stroke.png"
    else
        println("Cascade evolution")
        ρt = Matrix(kron(ρt, ρt))
        system_evolution = cascade_evolution(thermalizationtime, ρt, α0, [C, Cp, S, Sd], ga, gb)
        figname = "phaseonium_stroke_cascade.png"
    end

    cavity_evolution = [cavity.length for _ in 1:length(system_evolution)]

    g = plot_temperature(system_evolution, cavity_evolution, α0)
    times = range(1, length(system_evolution))
    γ = ga / gb
    # Thermalization function
    f(x, γ) = (final_temperature - T_initial) * (1 - exp(-γ * (x - 1))) + T_initial
    plot!(g, times, f.(times, γ), label="Theoretical Temperature", lc=:red)
    display(g)
    println("Press Enter to continue...")
    readline()
    savefig(g, figname)

    return system_evolution
    
end

main(1, 10)
