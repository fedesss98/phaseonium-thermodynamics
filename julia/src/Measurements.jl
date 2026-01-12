"""
Various measures on the Cavity
"""
module Measurements

using LinearAlgebra
using LaTeXStrings

export temperature

"""
Gives the temperature of one quantum thermal Gibbs state
"""
function temperature(state, ω, args...)
    real(ω / log(state[1, 1] / state[2, 2]))
end


"""
Von Neumann entropy of a density matrix, given by:
-Tr(ρ log(ρ)) = - ∑ₙλₙlog(λₙ)
where λₙare the eigenvalues of the density matrix.

Copy-pasted from qojulia/QuantumOpticsBase.jl
"""
function entropy_vn(rho, args...; tol=1e-15) 
    evals::Vector{ComplexF64} = eigvals(rho)
    entr = zero(eltype(rho))
    for d ∈ evals
        if !(abs(d) < tol)
            entr -= d*log(d)
        end
    end
    return real(entr)
end


"""
Gives the expected value of photon's number in the Cavity
"""
function avg_number(state, ω, args...)
    dims = size(state)[1]
    N = Diagonal(0:dims-1)

    return tr(state * N)
end

"""
Gives the expected value of the energy of the cavity
"""
function avg_E(state, ω, args...)
    dims = size(state)[1]
    N = Diagonal(0:dims-1)
    
    return 0.5 * ω * tr(state * N)
end

function _trace(A, B)
    dot(A', B)
end


function _idd(ndims)
    Diagonal(ones(ndims))
end


function pressure(ρ, π, idd, α, l, S; s=0)

    coefficient = α / (2*l^2*S)
    # ω = α / l
    
    # Pressure Operator
    if s == 1
        op = kron(π, idd)
    elseif s == 2
        op = kron(idd, π)
    else
        op = π
    end

    real(coefficient * tr(ρ * op))
end


function pressure(ρ, π, α, l, S)

    coefficient = α / (2*l^2*S)

    ω = α / l

    real(coefficient * tr(ρ * π))
end


"""
PLOTTING
"""
function measure_and_plot(x, y, system_evolution, cavity_evolution, label; α=π, g=nothing, title=nothing)
    ys = []
    xs = []
    if x == "Entropy"
        x_measurement = Measurements.entropy_vn
        x_label = x
    elseif x == "Frequency"
        _frequency(ρ, ω) = ω
        x_measurement = _frequency
        x_label = L"\omega"
    end

    if y == "Energy"
        y_measurment = Measurements.avg_E
        y_label = y
    elseif y == "n"
        y_measurment = Measurements.avg_number
        y_label = L"\langle \hat{n} \rangle"
    elseif y == "Temperature"
        y_measurment = Measurements.temperature
        y_label = y
    end
    
    for (i, ρ) in enumerate(system_evolution)
        cavity_len = cavity_evolution isa Real ? cavity_evolution : cavity_evolution[i]
        local ω = α / cavity_len
        x = real(round(x_measurement(ρ, ω), digits=5))
        y = real(round(y_measurment(ρ, ω), digits=5))
        
        push!(xs, x)
        push!(ys, y)
    end

    if isnothing(g)
        g = plot(xs, ys, label=label)
    else
        plot!(g, xs, ys, label=label)
    end
        
    # Plot starting point
    scatter!(g, [xs[1]], [ys[1]], label="Start", mc="blue", ms=5, msw=0)
    # Plot ending point
    scatter!(g, [xs[end]], [ys[end]], label="End", mc="red", ms=2.5, msw=0)
    if isnothing(title)
        title!(label)
    else
        title!(title)
    end
    xlabel!(x_label)
    ylabel!(y_label)
    
    return g
end


function plot_strokes_overlays(g, ys, isochore_samplings, adiabatic_samplings; x_min=0, x_max=1000, y_min=nothing, y_max=nothing)

    function _rectangle(x, w, h_up, h_down)
        Shape([
                (x, h_down),
                (x, h_up),
                (x+w, h_up),
                (x+w, h_down)
        ])
    end
    
    heating_distance = 2 * (isochore_samplings+adiabatic_samplings)
    isochore_strokes = 1:heating_distance:length(ys)
    adiabatic_strokes = isochore_samplings+adiabatic_samplings:isochore_samplings+adiabatic_samplings:length(ys)
    if y_max === nothing
        y_max = maximum(ys) + 0.1 * maximum(ys)
    end
    if y_min === nothing    
        y_min = minimum(ys) > 0 ? minimum(ys) -0.1 * minimum(ys) : minimum(ys) + 0.1 * minimum(ys) 
    end
    for left in isochore_strokes
        plot!(g, _rectangle(left, isochore_samplings+1, y_max, y_min), fillcolor=:red, alpha=0.05, label=false)
        left_cooling = left+isochore_samplings+adiabatic_samplings+1
        plot!(g, _rectangle(left_cooling, isochore_samplings+1, y_max, y_min), fillcolor=:blue, alpha=0.05, label=false)
    end
    xlims!(x_min, x_max)
    ylims!(y_min, y_max)
end


function plot_in_time(observable, system_evolution, cavity_evolution, label, title; 
        g=nothing, α=π, isochore_samplings=1, adiabatic_samplings=1, x_max=1000)
    temperatures = []
    if observable == "n"
        measurement = Measurements.avg_number
        y_label = L"\langle \hat{n} \rangle"
    elseif observable == "T"
        measurement = Measurements.temperature
        y_label = "Temperature"
    end
    
    for (i, ρ) in enumerate(system_evolution)
        cavity_len = cavity_evolution isa Real ? cavity_evolution : cavity_evolution[i]
        local ω = α / cavity_len
        t = real(round(measurement(ρ, ω), digits=5))
        push!(temperatures, t)
    end

    if isnothing(g)
        # Compose the whole plot with overlayed strokes
        g = plot(temperatures, label=label)
        plot_strokes_overlays(g, temperatures, isochore_samplings, adiabatic_samplings)
    else
        plot!(temperatures, label=label)
    end
    
    title!(title)
    xlabel!("Time")
    ylabel!(y_label)

    return g
    
end


"""
==== END MODULE =====
"""
end

