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
    γ::Float64
    velocity::Float64
    acceleration::Float64
    expanding_force::Float64
    compressing_force::Float64

    # Inner constructor for full initialization
    function Cavity(mass::Real, surface::Real, length::Real, l_min::Real, l_max::Real, α::Real, 
                    γ::Real = 0.0, velocity::Real = 0.0,
                    acceleration::Real = 0.0, expanding_force::Real = 0.0, compressing_force::Real = 0.0)
        new(
            convert(Float64, mass), 
            convert(Float64, surface), 
            convert(Float64, length), 
            convert(Float64, l_min), 
            convert(Float64, l_max), 
            convert(Float64, α), 
            convert(Float64, γ),
            convert(Float64, velocity), 
            convert(Float64, acceleration), 
            convert(Float64, expanding_force),
            convert(Float64, compressing_force)
        )
    end
end

# Convenience constructor for static cavity starting from miminum length
Cavity(m, s, l_min, l_max, α, γ, exp::String, cmp::String) = 
    Cavity(m, s, l_min, l_min, l_max, α, γ, 0.0, 0.0, parse(Float64, exp), parse(Float64, cmp))

Cavity(m::Real, s::Real, l_min::Real, l_max::Real, α::Real, γ::Real, exp::Real, cmp::Real) = 
    Cavity(m, s, l_min, l_min, l_max, α, γ, 0.0, 0.0, exp, cmp)

end  # module
