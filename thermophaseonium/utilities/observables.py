"""
List of common observables on Density Matrices
"""
import numpy as np
import qutip as qt


def is_diagonal(rho):
    return qt.qdiags(rho.diag(), 0) == rho


def _temperature_from_photon_number(n, omega):
    kB = 1
    hbar = 1
    return (hbar * omega) / kB * 1 / np.log(1 + 1 / n)


def _temperature_gibbs_state(rho):
    kB = 1
    hbar = 1
    omega = rho.omega
    z = 1 / rho.diag()[0]
    return - 1 / np.log(rho.diag()[1] * z)


def _von_neumann_entropy(rho):
    """Von Neumann entropy of the system in base e"""
    # ent = -np.trace(rho * np.log(rho.full()))
    return qt.entropy_vn(rho)


def temperature(rho):
    """Temperature of the system"""
    if not rho.is_diagonal():
        return _temperature_from_photon_number(rho.n, rho.omega)
    else:
        z = 1 / rho.diag()[0]
        return - 1 / np.log(rho.diag()[1] * z)


def entropy(rho):
    """Entropy of the system"""
    return _von_neumann_entropy(rho)