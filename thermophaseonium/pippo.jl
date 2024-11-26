#=
pippo:
- Julia version: 1.9.4
- Author: fedes
- Date: 2024-09-04
=#
 using QuantumOptics

const NDIMS = 30
const L0 = 10.0
const V = 1e-2
const α0 = 2*pi
const T_c = 1.0
const T_h = 10.0
const STROKE_TIME = 1

println("Maser engine started")
println(round(ℯ, digits=3))

boltzmann_population(omega, temp) = 1 / (ℯ^(omega / temp) + 1)

# Maximum and minimum lengths reached by the cavity
max_len = L0 + V * STROKE_TIME
min_len = L0 - V * STROKE_TIME
# Frequencies of the cavity
ω_cold = α0 / max_len
ω_hot = α0 / min_len
# Populations of the cavity in the cold and hot baths
p_cold = boltzmann_population(ω_cold, T_c)
p_hot = boltzmann_population(ω_hot, T_h)
# Operators and Hamiltonian
basis = FockBasis(NDIMS)
a = destroy(basis)
at = create(basis)
n = number(basis)

println("Cavity frequencies: $ω_cold, $ω_hot")