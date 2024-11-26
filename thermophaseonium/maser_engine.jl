import QuantumOptics

const NDIMS = 30
const L0 = 10.0
const V = 1e-2
const α0 = 2*pi
const T_c = 2.0
const T_h = 5.0
const STROKE_TIME = 1

println("Maser engine started")

boltzmann_population(omega, temp) = 1 / (ℯ^(omega / temp) + 1)

# Maximum and minimum lengths reached by the cavity
max_len = L0 + V * STROKE_TIME
min_len = L0 - V * STROKE_TIME
# Frequencies of the cavity
ω = α0 / L0
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
H = ω * ( n + identityoperator(basis)*1/2 )

function hilbert_is_good(system)
    if system[-1] < 1e-6
        return true
    else
        return false
    end

system = thermalstate(H, T_c)
if not hilbert_is_good(system)
    throw(AssertionError("Initial state is not a good Hilbert state"))
end

println("Cavity frequencies: $ω_cold, $ω_hot")

for i in [1:TIME_STEPS]
    system = timeevolution.schroedinger(TIME_STEP, system, H)
    if hilbert_is_good(system)
        break
    end
end
