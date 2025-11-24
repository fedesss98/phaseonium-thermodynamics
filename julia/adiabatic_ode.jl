"""
Reference quantities
L:
1mm = 5068eV
Temperature:
10^3K = 0.0862eV
Frequency:
1MHz = 4.13e-9eV 
10GHz = 10000MHz = 4.13e-5
"""

using DifferentialEquations
using QuantumOptics
using LinearAlgebra
using LaTeXStrings
using Plots
includet("./modules/Phaseonium.jl")
includet("./modules/OpticalCavity.jl")
includet("./modules/BosonicOperators.jl")
using .Phaseonium
using .OpticalCavity
using .BosonicOperators
includet("src/RoutineFunctions.jl")

# Physical Constants (Normalized)
const ħ = 1.0
const α0 = 2*π
const m_piston = 1.0   # Mass
const surface = 5_068.0
const γ_damping = 0.0   # Friction, energy dissipation
const n_cutoff = 250
v_initial = 0
T_phaseonium = 0.3

# Create the Cavity
l_min = 5068.0
l_max = l_min + 0.01 * l_min  # 10% Expansion
expanding_force = 1e-12
compressing_force = 1e-5
F_ext = expanding_force
cavity = Cavity(m_piston, surface, l_min, l_max, α0, expanding_force, compressing_force)
ω = α0 / l_min

# Prepare the state
rho_initial = complex(thermalstate(n_cutoff, ω, T_phaseonium))
# Define Kraus Operators
a = BosonicOperators.destroy(n_cutoff)
ad = BosonicOperators.create(n_cutoff)

n = ad * a
const n_conserved = real(tr(n * rho_initial))

println("--- Simulation Setup ---")
println("Initial Length: $l_min")
println("Target Max Length: $l_max")
println("Conserved <n>: $n_conserved")

# ==============================================================================
# 2. DEFINING THE PHYSICS
# ==============================================================================

# A. The Equations of Motion (2 coupled ODEs)
# u[1] = L (Position)
# u[2] = v (Velocity)
# p[1] = <n> (Passed as parameter to ensure conservation)
function piston_dynamics!(du, u, p, t)
    L = u[1]
    v = u[2]
    avg_n = p[1]

    # Radiation Force Definition
    # F = -dE/dL = (ħ * α / L^2) * (<n> + 1/2)
    # The +0.5 is the vacuum energy contribution explicitly kept in Tejero [cite: 68]
    F_rad = (ħ * α0 / L^2) * (avg_n + 0.5)
    
    # Net Force
    F_net = F_rad - F_ext - γ_damping * v

    du[1] = v
    du[2] = F_net / m_piston
end

# B. The Termination Condition (Callback)
# This function triggers when it returns 0. 
# We want it to trigger when L(t) - L_max = 0.
condition(u, t, integrator) = u[1] - l_max

# What to do when triggered: Stop the integrator
affect!(integrator) = terminate!(integrator)

# Create the callback
# "continuous" means the solver will interpolate to find the EXACT time L hits L_max
cb = ContinuousCallback(condition, affect!)

# ==============================================================================
# 3. RUNNING THE EVOLUTION
# ==============================================================================

# Initial state vector for ODE [L, v]
u0 = [l_min, v_initial]

# Time span (Make it large enough to ensure we reach L_max, the callback will stop it early)
t_span = (0.0, 1000.0)

# Define the problem
prob = ODEProblem(piston_dynamics!, u0, t_span, [n_conserved])

# Solve with the callback
sol = solve(prob, Tsit5(), callback=cb, reltol=1e-8, abstol=1e-8)

println("Simulation finished. Final time: $(sol.t[end])")
println("Final Length: $(sol.u[end][1])")

# ==============================================================================
# 4. RECONSTRUCTING EVOLUTION OF VARIABLES
# ==============================================================================

# A. Classical Variables
time_steps = sol.t
L_evolution = [u[1] for u in sol.u]
v_evolution = [u[2] for u in sol.u]

# B. Density Matrix Evolution
# Since H(t) commutes with rho_initial (both are diagonal), 
# rho(t) is CONSTANT in the interaction picture / lab frame for populations.
# We create an array filled with the initial density matrix for every time step.
rho_evolution = [rho_initial for _ in time_steps]

# C. Pressure Evolution (Clapeyron Diagram Data)
# P = F_rad / S
P_evolution = [(ħ * α0 / L^2) * (n_conserved + 0.5) / surface for L in L_evolution]

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
p1 = plot(time_steps, L_evolution, label="Length L(t)", ylabel="Length", lw=2)
# Add a dashed line for L_max to verify
hline!([l_max], label="Target L_max", linestyle=:dash, color=:black)

p2 = plot(time_steps, P_evolution, label="Pressure P(t)", ylabel="Pressure", xlabel="Time", color=:red, lw=2)

# PV Diagram (Pressure vs Volume) - Note Volume = S * L
V_evolution = surface .* L_evolution
p3 = plot(V_evolution, P_evolution, label="Expansion Stroke", xlabel="Volume", ylabel="Pressure", legend=:topright, lw=2)

plot(p1, p2, p3, layout=(3,1), size=(600, 800))julia -e 'import Pkg; Pkg.activate(); Pkg.add("LanguageServer"); Pkg.instantiate();'
