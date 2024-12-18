"""
Variables associated to an Optical Cavity
"""
module OpticalCavity
export Cavity

"""
Represents an Optical Cavity with its physical properties
"""
mutable struct Cavity
    mass::Float64
    surface::Float64
    length::Float64
    α::Float64
    acceleration::Float64
    external_force::Float64

    # Inner constructor for full initialization
    function Cavity(mass::Real, surface::Real, length::Real, α::Real, 
                    acceleration::Real = 0.0, external_force::Real = 0.0)
        new(
            convert(Float64, mass), 
            convert(Float64, surface), 
            convert(Float64, length), 
            convert(Float64, α), 
            convert(Float64, acceleration), 
            convert(Float64, external_force)
        )
    end
end

# Convenience constructor for static cavity
Cavity(m::Real, s::Real, l::Real, α::Real, f::Real) = 
    Cavity(m, s, l, α, 0.0, f)

end  # module