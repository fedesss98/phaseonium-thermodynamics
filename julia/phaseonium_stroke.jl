"""
Try a phaseonium stroke
"""

using LinearAlgebra
using ProgressBars
using Plots

include("./src/RoutineFunctions.jl")
include("./modules/modules_interface.jl")
include("./modules/MasterEquations.jl")

import .BosonicOperators
import .Phaseonium
import .Measurements
import .Thermodynamics
using .OpticalCavity

function _check(ρ)
    println("System after the stroke:")
    println(checkdensity(ρ))
    println("Final Temperature of the System: $(Measurements.temperature(ρ, ω))")
end

NDIMS = 25
Ω = 1.0
Δt = 1e-2

T_initial = 1.0
T_final = 1.5

# Create a Cavity
α0 = 2*π
l0 = 1.0
cavity = Cavity(1.0, 1.0, l0, α0, 0, 0.05)
ω = α0 / l0

ρt = thermalstate(NDIMS, ω, T_initial)

# Create Phaseonium atoms
ϕ = π/2
α = Phaseonium.alpha_from_temperature(T_final, ϕ) 

η = Phaseonium.densitymatrix(α, ϕ)

ga, gb = Phaseonium.dissipationrates(α, ϕ)
println(
    "Apparent Temperature carried by Phaseonium atoms: 
    $(Phaseonium.finaltemperature(ga, gb))")

# Define Kraus Operators
a = BosonicOperators.destroy(NDIMS)
ad = BosonicOperators.create(NDIMS)

C = BosonicOperators.C(Ω*Δt, NDIMS)
Cp = BosonicOperators.Cp(Ω*Δt, NDIMS)
S = BosonicOperators.S(Ω*Δt, NDIMS)
Sd = BosonicOperators.Sd(Ω*Δt, NDIMS)

E0 = sqrt(1 - ga/2 - gb/2) * identity(NDIMS) 
E1 = sqrt(ga/2) * C
E2 = sqrt(ga) * S
E3 = sqrt(gb/2) * Cp
E4 = sqrt(gb) * Sd

kraus = [E0, E1, E2, E3, E4]

# Time Evolution loop
const THERMALIZATION_TIME = 200

#=function h(t, ndims)=#
#=    a = BosonicOperators.destroy(ndims)=#
#=    ad = BosonicOperators.create(ndims)=#
#=    return ad * a=#
#=end=#


system_evolution, cavity_evolution = Thermodynamics.adiabatic_stroke(
    ρt, THERMALIZATION_TIME, Δt, [a, ad], cavity; sampling_each=10)
#=system_evolution = Thermodynamics.phaseonium_stroke(ρt, THERMALIZATION_TIME, kraus; sampling_each=100)=#

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

println("Press something to exit")
quit = readline()

