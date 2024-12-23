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

"""
Implement the Phaseonium thermalization stroke for two cavities
"""
function phaseonium_stroke_2(ρ, time, bosonic_operators, ga, gb; sampling_steps=10, verbose=1)
    if verbose > 0
        println("Isochoric Stroke")
    end

    silent_evolution_time = div(time, sampling_steps)
    systems = Vector{typeof(ρ)}(undef, sampling_steps + 1)
    systems[1] = ρ  # Initialize with the input state

    # Tensor Bosonic Operators
    c, cp, s, sd = [convert(Matrix{ComplexF32}, b) for b in bosonic_operators]

    cc_2ssd = kron(c, c) - 2 * kron(s, sd)
    cs_scp = kron(c, s) + kron(s, cp)
    cpcp_2sds = kron(cp, cp) - 2 * kron(sd, s)
    cpsd_sdc = kron(cp, sd) + kron(sd, c)
    bosonic_operators = [cc_2ssd, cs_scp, cpcp_2sds, cpsd_sdc]
    
    # Temporal loop
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

    # Preallocate variables with reduced precision
    ρ = convert(SparseMatrixCSC{ComplexF32, Int64}, ρ)  # Convert to complex sparse
    systems = Vector{Matrix{ComplexF32}}(undef, sampling_steps + 1)
    systems[1] = ρ
    
    cavity_lengths = [cavity.length for _ in 0:sampling_steps]

    a, ad = jumps
    # Reduce precision
    a = convert(SparseMatrixCSC{Float32, Int64}, a)
    ad = convert(SparseMatrixCSC{Float32, Int64}, ad)

    # Pressure Operator
    # Decomposed in three parts (constant and rotating)
    n = ad * a
    π_a = a * a
    π_ad = ad * ad
    π_parts = (n, π_a, π_ad)

    # Preallocate variables with reduce precision
    π = spzeros(ComplexF32, size(n)...)
    h = spzeros(Float32, size(ρ)...)
    U = spzeros(ComplexF32, size(ρ)...)
    
    alloc = (h, U, π)

    # Temporal loop
    verbose > 2 ? iter = ProgressBar(1:sampling_time) : iter=1:sampling_steps
    for i in iter
        ρ, cavity = MasterEquations.adiabaticevolve(
            ρ, Δt, silent_evolution_time, alloc, π_parts, cavity
        )
        systems[i+1] = ρ
        cavity_lengths[i+1] = cavity.length
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

    # Preallocate variables with reduced precision
    ρ = convert(SparseMatrixCSC{ComplexF32, Int64}, ρ)  # Convert to complex sparse
    systems = Vector{Matrix{ComplexF32}}(undef, sampling_steps + 1)
    systems[1] = ρ
    
    c1, c2 = cavities
    cavity_lengths = [[c1.length, c2.length] for _ in 0:sampling_steps]

    a, ad = jumps
    # Reduce precision
    a = convert(SparseMatrixCSC{Float32, Int64}, a)
    ad = convert(SparseMatrixCSC{Float32, Int64}, ad)

    # Pressure Operator
    # Decomposed in three parts (constant and rotating)
    n = ad * a
    π_a = a * a
    π_ad = ad * ad
    π_parts = (n, π_a, π_ad)

    # Preallocate variables with reduce precision
    π₁, π₂ = spzeros(ComplexF32, size(n)...), spzeros(ComplexF32, size(n)...)
    h = spzeros(Float32, size(ρ)...)
    U = spzeros(ComplexF32, size(ρ)...)
    alloc = (h, U, π₁, π₂)

    # Temporal loop
    verbose > 2 ? iter = ProgressBar(1:sampling_steps) : iter=1:sampling_steps
    for i in iter
        ρ, c1, c2 = MasterEquations.adiabaticevolve_2(
            ρ, [c1, c2], Δt, silent_evolution_time, alloc, π_parts
        )
        systems[i+1] = ρ
        cavity_lengths[i+1] = [c1.length, c2.length]
    end

    if verbose > 1
        # Show evolution of cavities' lenghts
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

