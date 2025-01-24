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
    l_min::Float64
    l_max::Float64
    α::Float64
    acceleration::Float64
    external_force::Float64

    # Inner constructor for full initialization
    function Cavity(mass::Real, surface::Real, length::Real, l_min::Real, l_max::Real, α::Real, 
                    acceleration::Real = 0.0, external_force::Real = 0.0)
        new(
            convert(Float64, mass), 
            convert(Float64, surface), 
            convert(Float64, length), 
            convert(Float64, l_min), 
            convert(Float64, l_max), 
            convert(Float64, α), 
            convert(Float64, acceleration), 
            convert(Float64, external_force)
        )
    end
end

# Convenience constructor for static cavity starting from miminum length
Cavity(m::Real, s::Real, l_min::Real, l_max::Real, α::Real, f::Real) = 
    Cavity(m, s, l_min, l_min, l_max, α, 0.0, f)

end  # module