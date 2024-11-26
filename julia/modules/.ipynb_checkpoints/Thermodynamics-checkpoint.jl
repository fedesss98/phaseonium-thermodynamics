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
        ρ_f = MasterEquations.krausevolve(
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

# function cycle(ρ, cavity, kraus thermalization_time, free_time,
# 
# end

end

