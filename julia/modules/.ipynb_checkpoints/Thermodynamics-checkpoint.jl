"""
Implement transformations for Thermodynamic strokes in a cycle
"""
module Thermodynamics

include("./MasterEquations.jl")

using Plots
using LaTeXStrings
using ProgressBars
using .MasterEquations


function phaseonium_stroke(ρ, time, kraus; sampling_steps=10, verbose=1)
    if verbose > 0
        println("Isochoric Stroke")
    end
    silent_evolution_time = div(time, sampling_steps) 
    
    global ρ_i = ρ
    systems = [Matrix(ρ_i) for _ in 0:sampling_steps]

    verbose > 2 ? iter = ProgressBar(1:sampling_steps) : iter=1:sampling_steps
    for i in iter
        ρ_f = MasterEquations.krausevolve_multithread(
            ρ_i, kraus, silent_evolution_time
        )
        systems[i+1] = ρ_f  
        global ρ_i = ρ_f
    end

    if verbose > 1
        g = plot(
            0:sampling_steps, [real(r[2,2]/r[1,1]) for r in systems], 
            label=L"\rho_{11} / \rho_{00}",
            title="Phaseonium Stroke"
        )
        display(g)
    end
    
    return systems
end

function phaseonium_stroke_2(ρ, time, bosonic_operators, ga, gb; sampling_steps=10, verbose=1)
    if verbose > 0
        println("Isochoric Stroke")
    end

    silent_evolution_time = div(time, sampling_steps)
    systems = Vector{typeof(ρ)}(undef, sampling_steps + 1)
    systems[1] = ρ  # Initialize with the input state

    iter = verbose > 2 ? ProgressBar(1:sampling_steps) : 1:sampling_steps
    for i in iter
        ρ = MasterEquations.meqevolve(ρ, bosonic_operators, ga, gb, silent_evolution_time)
        systems[i + 1] = ρ
    end

    if verbose > 1
        ratios = [real(r[2, 2] / r[1, 1]) for r in systems]
        g = plot(
            0:sampling_steps, ratios,
            label=L"\rho_{11} / \rho_{00}",
            title="Phaseonium Stroke"
        )
        display(g)
    end

    return systems
end


function adiabatic_stroke(ρ, time::Int64, Δt::Float64, jumps, cavity; sampling_steps=10, verbose=1)
    if verbose > 0
        println("Adiabatic Stroke")
    end
    silent_evolution_time = div(time, sampling_steps)

    global ρ_i = ρ
    systems = Vector{Matrix{ComplexF64}}(undef, sampling_steps + 1)
    systems[1] = ρ_i
    
    cavity_lengths = [cavity.length for _ in 0:sampling_steps]

    verbose > 2 ? iter = ProgressBar(1:sampling_time) : iter=1:sampling_steps
    for i in iter
        ρ_f, cavity = MasterEquations.adiabaticevolve(
            ρ_i, Δt, silent_evolution_time, jumps, cavity
        )
        systems[i+1] = ρ_f
        cavity_lengths[i+1] = cavity.length
        global ρ_i = ρ_f
    end

    if verbose > 1
        g = plot(
            0:sampling_steps, cavity_lengths, 
            label="Cavity Length",
            title="Adiabatic Stroke",
        )
        display(g)
    end
    
    return systems, cavity_lengths
end


function adiabatic_stroke_2(ρ, cavities, time::Int64, Δt::Float64, jumps; sampling_steps=10, verbose=1)
    if verbose > 0
        println("Adiabatic Stroke")
    end
    silent_evolution_time = div(time, sampling_steps)

    systems = Vector{Matrix{ComplexF64}}(undef, sampling_steps + 1)
    systems[1] = ρ
    
    c1, c2 = cavities
    cavity_lengths = [[c1.length, c2.length] for _ in 0:sampling_steps]
    verbose > 2 ? iter = ProgressBar(1:sampling_steps) : iter=1:sampling_steps
    for i in iter
        ρ, c1, c2 = MasterEquations.adiabaticevolve_2(
            ρ, [c1, c2], Δt, silent_evolution_time, jumps
        )
        systems[i+1] = ρ
        cavity_lengths[i+1] = [c1.length, c2.length]
    end
    if verbose > 1
        c1_lengths = [l1 for (l1, l2) in cavity_lengths]
        c2_lengths = [l2 for (l1, l2) in cavity_lengths]
        g = plot(
            0:sampling_steps, c1_lengths, 
            label="Cavity 1 Length",
            title="Adiabatic Stroke",
        )
        plot!(0:sampling_steps, c2_lengths, label="Cavity 2 Length")
        display(g)
    end
    
    return systems, cavity_lengths
end

# function cycle(ρ, cavity, kraus thermalization_time, free_time,
# 
# end

end

