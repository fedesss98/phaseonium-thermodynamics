
"""
Implement transformations for Thermodynamic strokes in a cycle
"""
module Thermodynamics

include("./MasterEquations.jl")

using Plots
using LaTeXStrings
using ProgressBars
using DifferentialEquations
using LinearAlgebra
using SparseArrays
using .MasterEquations


function phaseonium_stroke(ρ, time, bosonic_operators, coefficients; sampling_steps=10, verbose=1)
    if verbose > 0
        println("Isochoric Stroke")
    end
    silent_evolution_time = div(time, sampling_steps) 
    ga, gb = coefficients
    ndims = size(ρ)[1]
    systems = [Matrix(ρ) for _ in 0:sampling_steps]

    verbose > 1 ? iter = ProgressBar(1:sampling_steps) : iter=1:sampling_steps
    for i in iter
        ρ = MasterEquations.meqevolve(
            ρ, bosonic_operators, ga, gb, silent_evolution_time, ndims
        )
        systems[i+1] = ρ
    end

    if verbose > 2
        g = plot(
            0:sampling_steps, [real(r[2,2]/r[1,1]) for r in systems], 
            label=L"\rho_{11} / \rho_{00}",
            title="Phaseonium Stroke"
        )
        display(g)
    end
    
    return systems
end

function phaseonium_stroke_2(ρ, time, bosonic_operators, ga, gb; sampling_steps=10, verbose=1, io=nothing)
    if verbose > 0
        println(io, "Isochoric Stroke")
    end

    # Preallocate Tensor Bosonic Operators
    c, cp, s, sd = bosonic_operators

    cc_2ssd = kron(c, c) - 2 * kron(s, sd)
    cs_scp = kron(c, s) + kron(s, cp)
    cpcp_2sds = kron(cp, cp) - 2 * kron(sd, s)
    cpsd_sdc = kron(cp, sd) + kron(sd, c)
    tensor_bosonic_opearators = cc_2ssd, cs_scp, cpcp_2sds, cpsd_sdc

    ndims = size(ρ)[1]    
    systems = Vector{typeof(ρ)}(undef, sampling_steps)
    systems[1] = ρ  # Initialize with the input state

    silent_evolution_time = div(time, sampling_steps)
    
    iter = verbose > 2 ? ProgressBar(2:sampling_steps) : 2:sampling_steps
    for i in iter
        ρ = MasterEquations.meqevolve_2(ρ, tensor_bosonic_opearators, ga, gb, silent_evolution_time, ndims)
        systems[i] = ρ
    end

    if verbose > 1
        ratios = [real(r[2, 2] / r[1, 1]) for r in systems]
        g = plot(
            1:sampling_steps, ratios,
            label=L"\rho_{11} / \rho_{00}",
            title="Phaseonium Stroke"
        )
        display(g)
    end

    return systems
end


function adiabatic_stroke(ρ, time::Int64, Δt::Float64, jumps, cavity, idd, π_parts; sampling_steps=10, verbose=1)
    if verbose > 0
        println("Adiabatic Stroke")
    end
    silent_evolution_time = div(time, sampling_steps)

    global ρ_i = ρ
    systems = Vector{Matrix{ComplexF64}}(undef, sampling_steps + 1)
    systems[1] = ρ_i
    
    cavity_lengths = [cavity.length for _ in 0:sampling_steps]

    verbose > 2 ? iter = ProgressBar(1:sampling_steps) : iter=1:sampling_steps
    for i in iter
        ρ_f, cavity = MasterEquations.adiabaticevolve(
            ρ_i, Δt, silent_evolution_time, jumps, cavity, idd, π_parts
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

"""
Allocate variables---use SparseMatrices where possible
"""
function make_cache_adiabatic_1(jumps, force)
    opa, opad = jumps
    n = opad * opa
    dims = size(n)[1]
    identity_matrix = spdiagm(ones(dims))

    return (;
        U = spzeros(ComplexF64, dims, dims),
        Ud = spzeros(ComplexF64, dims, dims),
        temp = zeros(ComplexF64, dims, dims),
        idd = identity_matrix,
        n = n,
        π_a = opa * opa,
        π_ad = opad * opad,
        π_op = Matrix{ComplexF64}(undef, dims, dims),
        _h = sparse(n .+ 0.5 .* identity_matrix),
        h1 = spzeros(ComplexF64, dims, dims),
        a1 = 0,
        p1 = 0,
        force = force,
    )
end

function make_cache_adiabatic_2(jumps, forces)
    opa, opad = jumps
    n = opad * opa
    dims = size(n)[1]
    identity_matrix = spdiagm(ones(dims))

    return (;
        U = spzeros(ComplexF64, dims^2, dims^2),
        Ud = spzeros(ComplexF64, dims^2, dims^2),
        temp = zeros(ComplexF64, dims^2, dims^2),
        idd = identity_matrix,
        n = n,
        π_a = opa * opa,
        π_ad = opad * opad,
        π_op = Matrix{ComplexF64}(undef, dims, dims),
        _h = sparse(n .+ 0.5 .* identity_matrix),
        h1 = spzeros(ComplexF64, dims, dims),
        h2 = spzeros(ComplexF64, dims, dims),
        h = spzeros(ComplexF64, dims^2, dims^2),
        a1 = 0,
        a2 = 0,
        p1 = 0,
        p2 = 0,
        force1 = forces[1],
        force2 = forces[2],
    )
end


"""
function _adiabatic_stroke_2(ρ, cavities, time::Int64, Δt::Float64, jumps, process; sampling_steps=10, verbose=1)
    if verbose > 0
        println("Adiabatic Stroke")
    end
    silent_evolution_time = div(time, sampling_steps)

    # Operators are defined on one subspace
    if process == "Expansion"
        forces = [cavity.expanding_force for cavity in cavities]
    else
        forces = [cavity.compressing_force for cavity in cavities]
    end
    dims = Int(sqrt(size(ρ)[1]))
    cache = make_cache_adiabatic(dims, forces)
    
    systems = Vector{Matrix{ComplexF64}}(undef, sampling_steps + 1)
    systems[1] = ρ
    
    c1, c2 = cavities
    # Cavities start fixed
    c1.acceleration = 0
    c2.acceleration = 0
    cavity_lengths = [[c1.length, c2.length] for _ in 0:sampling_steps]
    verbose > 2 ? iter = ProgressBar(1:sampling_steps) : iter=1:sampling_steps
    for i in iter
        ρ, c1, c2 = MasterEquations.adiabaticevolve_2(
            ρ, [c1, c2], Δt, i, silent_evolution_time, alloc, π_parts
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
"""


function adiabatic_stroke_1(ρ, cavity, jumps, Δt::Float64, process; sampling_steps=10, verbose=1, io=nothing)

    expansion_process = process == "Expansion"
    # Select force depending on process
    force = expansion_process ? cavity.expanding_force : cavity.compressing_force
    cache = make_cache_adiabatic_1(jumps, force)  # still pass as vector if cache expects it

    # Initialize system tracking
    systems = Vector{Matrix{ComplexF64}}(undef, sampling_steps)
    systems[1] = copy(ρ)

    # Setup cavity
    cavity.acceleration = 0  # starts blocked

    # Determine expansion/contraction direction
    l_start = expansion_process ? cavity.l_min : cavity.l_max
    l_end   = expansion_process ? cavity.l_max : cavity.l_min
    direction = expansion_process ? 1 : -1
    l_samplings = range(l_start, stop=l_end, length=sampling_steps)
    
    cavity_lengths = Vector{Float64}(undef, sampling_steps)
    cavity_lengths[1] = cavity.length
    
    verbose > 0 && println(io, "Adiabatic $process")
    verbose > 2 && (iter = ProgressBar(total=sampling_steps))
    
    t = 0.0
    i = 2
    stop = false

    while i <= sampling_steps
        if direction * cavity.length >= direction * l_samplings[i]
            systems[i] = copy(ρ)
            cavity_lengths[i] = cavity.length
            verbose > 2 && update(iter)
            i += 1
        elseif i > sampling_steps
            stop = true
        end
        
        ρ, cavity = MasterEquations.adiabaticevolve_1(
            ρ, cavity, cache, Δt, t, process, stop
        )
        t += Δt

        verbose > 0 && i % 10 == 0 && println(io, "force: $force, acceleration: $(cavity.acceleration)")

    end
    
    if verbose > 1
        g = plot(
            1:sampling_steps, cavity_lengths, 
            label="Cavity Length",
            title="Adiabatic Stroke",
        )
        display(g)
    end
    
    return systems, cavity_lengths, t
end



function adiabatic_stroke_2(ρ, cavities, jumps, Δt::Float64, process; sampling_steps=10, verbose=1, io=nothing)

    # Allocate variables in the cache
    forces = process == "Expansion" ? [cavity.expanding_force for cavity in cavities] : [cavity.compressing_force for cavity in cavities]
    cache = make_cache_adiabatic_2(jumps, forces)

    # Initialize system tracking
    systems = Vector{Matrix{ComplexF64}}(undef, sampling_steps)
    systems[1] = ρ

    # Setup cavities
    c1, c2 = cavities
    c1.acceleration = c2.acceleration = 0  # They start blocked

    # Determine expansion/contraction direction
    l_start = process == "Expansion" ? c1.l_min : c1.l_max
    l_end = process == "Expansion" ? c1.l_max : c1.l_min
    direction = process == "Expansion" ?  1 : -1
    l_samplings = collect(range(l_start, stop=l_end, length=sampling_steps))
    
    cavity_lengths = [[c1.length, c2.length] for _ in 1:sampling_steps]
    
    if verbose > 0
        println(io, "Adiabatic $process")
    end

    if verbose > 2
        iter = ProgressBar(total=sampling_steps)
    end
    
    t = 0
    i = j = 2
    stop1 = stop2 = false
    while i <= sampling_steps || j <= sampling_steps
        
        if i <= sampling_steps && direction * c1.length >= direction * l_samplings[i]
            systems[i] = ρ
            cavity_lengths[i] = [c1.length, c2.length]
            verbose > 2 && update(iter)
            i += 1
        elseif i > sampling_steps
            stop1 = true
        end
        if j <= sampling_steps && direction * c2.length >= direction * l_samplings[j]
            cavity_lengths[j] = [c1.length, c2.length]
            j += 1
        elseif j > sampling_steps
            stop2 = true
        end
        ρ, c1, c2 = MasterEquations.adiabaticevolve_2(
            ρ, (c1, c2), cache, Δt, t, process, stop1, stop2
        )
        t += Δt

        if verbose > 0 && i % 10 == 0
            force1, force2 = forces
            println(io, "f1:$force1 - f2:$force2\na1:$(c1.acceleration) - a2:$(c2.acceleration)")
        end
        
    end
    
    if verbose > 1
        c1_lengths = [l1 for (l1, _) in cavity_lengths]
        c2_lengths = [l2 for (_, l2) in cavity_lengths]
        g = plot(
            1:sampling_steps, c1_lengths, 
            label="Cavity 1 Length",
            title="Adiabatic Stroke",
        )
        plot!(1:sampling_steps, c2_lengths, label="Cavity 2 Length")
        display(g)
    end
    
    return systems, cavity_lengths, t
end


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
    α = p[3]
    m = p[4]
    γ_damping = p[5]

    # Radiation Force
    # F_rad = -dE/dL = (ħ * α / L^2) * (<n> + 1/2)
    # The +0.5 accounts for vacuum energy pressure
    F_rad = α / L^2 * (avg_n + 0.5)

    # Net Force (including friction)
    F_net = F_rad - F_ext - γ_damping * v

    du[1] = v
    du[2] = F_net / m
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
    sampling_steps=100.0, max_time=1000.0, verbose=1, io=nothing)


    # Determine Direction and Target
    l0 = cavity.length
    v0 = cavity.velocity
    if l0 == cavity.l_min
        target_l = cavity.l_max
        direction = 1.0 # Expanding
        external_force = cavity.expanding_force
        process = "Expansion"
    else
        target_l = cavity.l_min
        direction = -1.0 # Compressing
        external_force = cavity.compressing_force
        process = "Compression"
    end

    if verbose > 0
        println(io, "Adiabatic $process")
    end

    # Stop exactly when L crosses target_l
    condition(u, t, integrator) = u[1] - target_l
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    # Setup ODE Problem
    u0 = [l0, v0]
    γ_damping = cavity.γ
    p = [avg_n, external_force, cavity.α, cavity.mass, γ_damping]
    t_span = (0.0, max_time)

    prob = ODEProblem(piston_ode!, u0, t_span, p)

    # Solve with high tolerance for precision
    sol = solve(prob, Tsit5(), callback=cb, reltol=1e-9, abstol=1e-9)

    # Interpolate Output (Sampling)
    # Determine output time points
    t_end = sol.t[end]
    dt = 1.0 / sampling_steps
    t_eval = collect(0.0:dt:t_end)
    
    # Ensure the exact final point is included
    if t_eval[end] != t_end
        push!(t_eval, t_end)
    end

    # Evaluate solution at sampling points
    u_eval = sol(t_eval)
    
    # Reconstruct Outputs
    cavity_lengths = [u[1] for u in u_eval.u]
    cavity_velocities = [u[2] for u in u_eval.u]
    # Reconstruct States
    systems = [rho_initial for _ in t_eval]

    if verbose > 1
        g = plot(t_eval, cavity_lengths,
                 label="Cavity Lengths", title="Adiabatic $process")
        display(g)
    end


    return systems, cavity_lengths, cavity_velocities, t_eval[end]
end



# function cycle(ρ, cavity, kraus thermalization_time, free_time,
# 
# end

end

