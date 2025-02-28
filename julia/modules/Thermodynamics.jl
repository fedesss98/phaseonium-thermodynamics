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


function phaseonium_stroke(ρ, time, kraus; sampling_steps=10, verbose=1)
    if verbose > 0
        println("Isochoric Stroke")
    end
    silent_evolution_time = div(time, sampling_steps) 
    
    systems = [Matrix(ρ) for _ in 0:sampling_steps]

    verbose > 2 ? iter = ProgressBar(1:sampling_steps) : iter=1:sampling_steps
    for i in iter
        ρ = MasterEquations.krausevolve(
            ρ, kraus, silent_evolution_time
        )
        systems[i+1] = ρ
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


function _adiabatic_stroke_2(ρ, cavities, time::Int64, Δt::Float64, jumps; sampling_steps=10, verbose=1)
    if verbose > 0
        println("Adiabatic Stroke")
    end
    silent_evolution_time = div(time, sampling_steps)

    # Operators are defined on one subspace
    dims = Int(sqrt(size(ρ)[1]))
    identity_matrix = spdiagm(ones(dims))
    opa, opad = jumps
    # Reduce precision
    opa = convert(SparseMatrixCSC{Float32, Int64}, opa)
    opad = convert(SparseMatrixCSC{Float32, Int64}, opad)
    # Pressure Operator
    # Decomposed in three parts (constant and rotating)
    n = opad * opa
    π_a = opa * opa
    π_ad = opad * opad
    π_parts = (n, π_a, π_ad) 

    # Preallocate variables with reduce precision
    π₁, π₂ = spzeros(ComplexF32, size(n)...), spzeros(ComplexF32, size(n)...)
    U = spzeros(ComplexF32, size(ρ)...)
    # ρ = convert(SparseMatrixCSC{ComplexF32, Int64}, ρ)  # Convert real sparse matrix to complex sparse
    alloc = (U, identity_matrix)
    
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


function adiabatic_stroke_2(ρ, cavities, Δt::Float64, jumps; sampling_steps=10, verbose=1)

    # Operators are defined on one subspace
    dims = Int(sqrt(size(ρ)[1]))
    identity_matrix = spdiagm(ones(dims))
    opa, opad = jumps
    # Reduce precision
    opa = convert(SparseMatrixCSC{Float32, Int64}, opa)
    opad = convert(SparseMatrixCSC{Float32, Int64}, opad)
    # Pressure Operator
    # Decomposed in three parts (constant and rotating)
    n = opad * opa
    π_a = opa * opa
    π_ad = opad * opad
    π_parts = (n, π_a, π_ad) 

    # Preallocate variables with reduce precision
    U = spzeros(ComplexF32, size(ρ)...)
    alloc = (U, identity_matrix)

    # Initialize system tracking
    systems = Vector{Matrix{ComplexF64}}(undef, sampling_steps)
    systems[1] = ρ

    # Setup cavities
    c1, c2 = cavities
    c1.acceleration = c2.acceleration = 0  # They start blocked

    # Determine expansion/contraction direction
    is_expanding = c1.l_min == c1.length
    l_start, l_end, direction = is_expanding ? (c1.l_min, c1.l_max, 1) : (c1.l_max, c1.l_min, -1)
    l_samplings = collect(range(l_start, stop=l_end, length=sampling_steps))
    
    cavity_lengths = [[c1.length, c2.length] for _ in 1:sampling_steps]
    
    if verbose > 0
        process = is_expanding ? "Expansion" : "Contraction"
        println("Adiabatic $process")
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
            ρ, [c1, c2], Δt, t, alloc, π_parts, process, stop1, stop2
        )
        t += Δt

        if i % 10 == 0
            if process == "Expansion"
                force1 = c1.expanding_force
                force2 = c2.expanding_force
            else
                force1 = c1.compressing_force
                force2 = c2.compressing_force
            end
            println("f1:$force1 - f2:$force2\na1:$(c1.acceleration) - a2:$(c2.acceleration)")
        end
        
    end
    
    if verbose > 1
        c1_lengths = [l1 for (l1, l2) in cavity_lengths]
        c2_lengths = [l2 for (l1, l2) in cavity_lengths]
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

