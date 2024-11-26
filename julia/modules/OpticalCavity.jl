"""
Variables associated to an Optical Cavity
"""
module OpticalCavity

export Cavity

mutable struct Cavity
    mass::Float32
    surface::Float32
    length::Float32
    α::Float32

    acceleration::Float32
    external_force::Float32
end

"""
Constructor to initialize a static Cavity
"""
Cavity(m::Real, s::Real, l::Real, α::Real, f::Real) = Cavity(Float32(m), Float32(s), Float32(l), Float32(α), Float32(0.0), Float32(f))


end
