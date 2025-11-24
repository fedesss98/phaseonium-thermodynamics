"""
Reference quantities
L:
1mm = 5068eV
Temperature:
10^3K = 0.0862eV
Frequency:
1MHz = 4.13e-9eV 
10GHz = 10000MHz = 
"""

using Expokit
using LaTeXStrings
using LinearAlgebra
using MKL
using Plots
using ProgressBars
using Revise
using SparseArrays
includet("./modules/Phaseonium.jl")
includet("./modules/OpticalCavity.jl")
includet("./modules/BosonicOperators.jl")
using .Phaseonium
using .OpticalCavity
using .BosonicOperators
includet("src/RoutineFunctions.jl")

ndims = 200
Ω = 1.0
Δt = 1e-2

T_initial = 0.01
T_final = 0.3

# Create a Cavity
α0 = 2*π
surface = 5_068.0
l_min = 506.8
l_max = l_min + 0.01 * l_min  # 10% Expansion
acceleration = 0.0
expanding_force = 1e-12
compressing_force = 1e-5
cavity = Cavity(1.0, surface, l_min, l_max, α0, expanding_force, compressing_force)
ω = α0 / l_min
println("Cavity frequency: $ω")

# We start with a thermalized cavity
global ρt = thermalstate(ndims, ω, T_final)
println("Cavity field last diagonal element: $(ρt[end])")

# Create Phaseonium atoms
ϕ = π/1.8
α = Phaseonium.alpha_from_temperature(T_final, ϕ, ω)
println("Phaseonium α: $α < $(sqrt((1+cos(ϕ))/(3+cos(ϕ))))")

η = Phaseonium.densitymatrix(α, ϕ)

ga, gb = Phaseonium.dissipationrates(α, ϕ)
final_temperature = Phaseonium.finaltemperature(ω, ga, gb)
println(
    "Apparent Temperature carried by Phaseonium atoms: 
    $(final_temperature)")

# Define Kraus Operators
a = BosonicOperators.destroy(ndims)
ad = BosonicOperators.create(ndims)

C = BosonicOperators.C(Ω*Δt, ndims)
Cp = BosonicOperators.Cp(Ω*Δt, ndims)
S = BosonicOperators.S(Ω*Δt, ndims)
Sd = BosonicOperators.Sd(Ω*Δt, ndims)

E0 = sqrt(1 - ga/2 - gb/2) * identity(ndims)
E1 = sqrt(ga/2) * C
E2 = sqrt(ga) * S
E3 = sqrt(gb/2) * Cp
E4 = sqrt(gb) * Sd

kraus = [E0, E1, E2, E3, E4]
jumps = (a, ad)



ρ = complex(thermalstate(ndims, ω, T_final))
n = ad * a
identity_matrix = spdiagm(ones(ndims))

cache = (;
        U = spzeros(ComplexF64, ndims, ndims),
        Ud = spzeros(ComplexF64, ndims, ndims),
        temp = zeros(ComplexF64, ndims, ndims),
        temp2 = zeros(ComplexF64, ndims, ndims),
        idd = identity_matrix,
        n = n,
        π_a = a * a,
        π_ad = ad * ad,
        π_op = Matrix{ComplexF64}(undef, ndims, ndims),
        _h = sparse(n .+ 0.5 .* identity_matrix),
        h1 = spzeros(ComplexF64, ndims, ndims),
        a1 = 0,
        p1 = 0,
        force = cavity.expanding_force,
)


function pressure(ρ, π, idd, α, l, S; s=0)

    coefficient = α / (2*l^2*S)
    op = π

    real(coefficient * tr(ρ * op))
end


function adiabaticevolve_1(ρ, cavity, cache, Δt, t, process, stop)
    # Unpack cache variables (only those needed for one cavity)
    U      = cache.U
    Ud     = cache.Ud
    idd    = cache.idd
    _h     = cache._h
    h      = cache.h1
    n      = cache.n
    π_a    = cache.π_a
    π_ad   = cache.π_ad
    π_op   = cache.π_op
    a1     = cache.a1
    p1     = cache.p1
    force = cache.force
    temp   = cache.temp

    α0  = cavity.α

    # Move the cavity wall if not stopped
    if !stop
        Δt² = Δt * Δt
        # Update the position within the boundaries of the cavity
        cavity.length = clamp(
            cavity.length + 0.5 * cavity.acceleration * Δt², 
            cavity.l_min, cavity.l_max)
    end

    # Update energy
    ω = α0 / cavity.length
    @. h = ω * _h

    # Evolve the system
    U  .= Expokit.padm(-im * h)
    Ud .= U'
    mul!(temp, U, ρ)
    mul!(ρ, temp, Ud)

    # Update pressure and acceleration
    two_ωt = 2 * ω * t
    exp_neg = exp(-im * two_ωt)
    exp_pos = exp(im * two_ωt)
    @. π_op = (2 * n + idd) - (π_a * exp_neg) - (π_ad * exp_pos)
    p1 = pressure(ρ, π_op, idd, α0, cavity.length, cavity.surface)

    a1 = (p1 * cavity.surface - force) / cavity.mass

    # Safety checks (consider removing in production for speed)
    @assert norm(a1) > 1e-12 "Cavity $(cavity.surface)m² is almost still during $process"
    if process == "Expansion"
        @assert a1 >= 0 "Cavity is going backward during expansion!"
    else
        @assert a1 <= 0 "Cavity is going forward during contraction!"
    end
    @assert cavity.acceleration * a1 >= 0 "Cavity changed direction during $process!"

    cavity.acceleration = a1

    return ρ, cavity
end


function adiabatic_stroke(ρ, cavity, jumps, cache, sampling_steps)
  process = "Expansion"

  force = cavity.expanding_force

  # Initialize system tracking
  systems = Vector{Matrix{ComplexF64}}(undef, sampling_steps)
  systems[1] = ρ

  # Setup cavity
  cavity.acceleration = 0  # starts blocked

  # Determine expansion/contraction direction
  l_start = cavity.l_min
  l_end   = cavity.l_max
  direction = 1
  l_samplings = collect(range(l_start, stop=l_end, length=sampling_steps))

  cavity_lengths = [cavity.length for _ in 1:sampling_steps]

  iter = ProgressBar(total=sampling_steps)

  t = 0.0
  n_iterations = 0
  print_interval = 50000
  i = 2
  stop = false
  println("Starting printing output every $(print_interval)!")
  while i <= sampling_steps
    if direction * cavity.length >= direction * l_samplings[i]
        systems[i] = ρ
        cavity_lengths[i] = cavity.length
        update(iter)
        i += 1
    elseif i > sampling_steps
        stop = true
    end
    ρ, cavity = adiabaticevolve_1(
        ρ, cavity, cache, Δt, t, process, stop
        )
    t += Δt

    if n_iterations % print_interval == 0 
      println("$n_iterations] force: $force, acceleration: $(cavity.acceleration), length: $(cavity.length)")
    end
    n_iterations += 1
  end
end













