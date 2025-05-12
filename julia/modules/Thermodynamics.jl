"""
Implement transformations for Thermodynamic strokes in a cycle
"""
module Thermodynamics

include("./MasterEquations.jl")

using Plots
using LaTeXStrings
using ProgressBars
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
function make_cache_adiabatic(jumps, forces)
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

function adiabatic_stroke_2(ρ, cavities, jumps, Δt::Float64, process; sampling_steps=10, verbose=1, io=nothing)

    # Allocate variables in the cache
    forces = process == "Expansion" ? [cavity.expanding_force for cavity in cavities] : [cavity.compressing_force for cavity in cavities]
    cache = make_cache_adiabatic(jumps, forces)

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

# function cycle(ρ, cavity, kraus thermalization_time, free_time,
# 
# end

end

