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

    # Full Inner Constructor
    function Cavity(m, s, l, lmin, lmax, a, g, v, acc, f_exp, f_cmp)
        new(m, s, l, lmin, lmax, a, g, v, acc, f_exp, f_cmp)
    end
end

# Main Outer Constructor with Keyword Arguments
function Cavity(mass, surface, length, l_min, l_max, α; 
                γ=0.0, velocity=0.0, acceleration=0.0, 
                expanding_force=0.0, compressing_force=0.0)
    
    return Cavity(mass, surface, length, l_min, l_max, α, 
                  γ, velocity, acceleration, expanding_force, compressing_force)
end

# Convenience Constructor for "Static" start
function Cavity(m, s, l_min, l_max, α, γ, exp::Real, cmp::Real)
    # Note: we pass l_min twice because 'length' starts at 'l_min'
    return Cavity(m, s, l_min, l_min, l_max, α; 
                  γ=γ, expanding_force=exp, compressing_force=cmp)
end

# String-parsing version
function Cavity(m, s, l_min, l_max, α, γ, exp::String, cmp::String)
    return Cavity(m, s, l_min, l_max, α, γ, parse(Float64, exp), parse(Float64, cmp))
end

end  # module
