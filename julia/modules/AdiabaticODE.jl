
"""
Simulate the Adiabatic Stroke as a differential equation.
Consider (ideally) the number of photons in the cavity constant and the Density Matrix coherences null.
"""
module AdiabaticODE

using DifferentialEquations
using QuantumOptics
using LinearAlgebra


"""
    piston_ode!(du, u, p, t)

The system of differential equations for the adiabatic stroke.
u[1]: Length L
u[2]: Velocity v
p[1]: Conserved Photon Number <n>
p[2]: External force
p[3]: Cavity struct
"""
function piston_ode!(du, u, p, t)
    L = u[1]
    v = u[2]
    avg_n = p[1]
    F_ext = p[2]
    sys = p[3] # Unpack system parameters

    # Radiation Force
    # F_rad = -dE/dL = (ħ * α / L^2) * (<n> + 1/2)
    # The +0.5 accounts for vacuum energy pressure
    F_rad = sys.α0 / L^2 * (avg_n + 0.5)

    # Net Force (including friction)
    F_net = F_rad - F_ext - γ_damping * v

    du[1] = v
    du[2] = F_net / sys.mass
end


"""
    adiabatic_stroke_ode(rho_initial, avg_n, cavity; 
                        sampling_freq=100.0, max_time=1000.0)

Simulates the adiabatic evolution of the cavity from `l0` until it reaches
the geometric limit (`cavity.l_max` if expanding, `cavity.l_min` if compressing).

# Arguments
- `rho_initial`: The density matrix at the start of the stroke.
- `avg_n`: The expectation value of the number operator on the density matrix, constant throughout the adiabatic process.
- `cavity`: The `Cavity` object defining the limits and forces.

# Keywords
- `sampling_freq`: Number of data points to output per unit time.
- `max_time`: Safety cutoff for integration.

# Returns
- `times`: Vector of time points.
- `lengths`: Vector of cavity lengths L(t).
- `velocities`: Vector of piston velocities v(t).
- `states`: Vector of density matrices ρ(t).
"""
function adiabatic_stroke_ode(
    rho_initial::AbstractMatrix{<:Number}, 
    avg_n::Float64, cavity;
    sampling_freq=100.0, max_time=1000.0)


    # Determine Direction and Target
    l0 = cavity.length
    v0 = cavity.velocity
    if l0 == cavity.l_min
        target_l = cavity.l_max
        direction = 1.0 # Expanding
        external_force = cavity.expanding_force
    else
        target_l = cavity.l_min
        direction = -1.0 # Compressing
        external_force = cavity.compressing_force
    end

    # Stop exactly when L crosses target_l
    condition(u, t, integrator) = u[1] - target_l
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    # Setup ODE Problem
    u0 = [l0, v0]
    p = [n_conserved, external_force, cavity]
    t_span = (0.0, max_time)

    prob = ODEProblem(piston_ode!, u0, t_span, p)

    # Solve with high tolerance for precision
    sol = solve(prob, Tsit5(), callback=cb, reltol=1e-9, abstol=1e-9)

    # Interpolate Output (Sampling)
    # Determine output time points
    t_end = sol.t[end]
    dt = 1.0 / sampling_freq
    t_eval = collect(0.0:dt:t_end)
    
    # Ensure the exact final point is included
    if t_eval[end] != t_end
        push!(t_eval, t_end)
    end

    # Evaluate solution at sampling points
    u_eval = sol(t_eval)
    
    # Reconstruct Outputs
    lengths = [u[1] for u in u_eval.u]
    velocities = [u[2] for u in u_eval.u]
    # Reconstruct States
    states = [rho_initial for _ in t_eval]

    return t_eval, lengths, velocities, states
end


end  # module
