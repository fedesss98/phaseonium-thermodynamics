"""
Functions to work with the Phaseonium
"""
module Phaseonium

export densitymatrix, dissipationrates, finaltemperature

"""
Gives the Density Matrix of one Phaseonium atom,
given the excited state population and the coherence phase
"""
function densitymatrix(α, ϕ)
    [
        α^2 0 0;
        0 (1 - α^2)/2 (1 - α^2)/2 * exp(ϕ*im);
        0 (1 - α^2)/2 * exp(-ϕ*im) (1 - α^2)/2
    ]
end


"""
Gives the rates of dissipation appearing in the Phaseonium Master Equation
"""
function dissipationrates(α, ϕ)
    ga = 2*α^2
    gb = (1 + cos(ϕ))*(1 - α^2)
    return real(ga), real(gb)
end


"""
Gives the stable apparent temperature carried by Phaseonium atoms
"""
function finaltemperature(ω, γα, γβ)
    return - ω / log(γα/γβ)
end


"""
Find the parameter ϕ that gives the Apparent Temperature specified, given α
"""
function phi_from_temperature(T, α, ω)
    T = T * ω
    return arccos(2*α^2 * exp(1/T) / (1-α^2) - 1)
end

"""
Find the parameter α that gives the Apparent Temperature specified, given ϕ
"""
function alpha_from_temperature(T, ϕ, ω)
    factor = (1+cos(ϕ)) / (2 * exp(ω/T) + 1 + cos(ϕ)) 
    return sqrt(factor)
end

end

