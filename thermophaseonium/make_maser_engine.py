import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
import tqdm
import pickle
from pathlib import Path

from thermophaseonium.utilities.states import Cavity, Ancilla

SAVE_PATH = Path(r"G:\UNIPA\DOTTORATO\PHASEONIUM\thermo-phaseonium\data")
NDIMS = 30
SIM_PARAMS = dict(
    l0=10.0,
    v=1e-2,
    alpha0=2*np.pi,
    T_c=2,
    T_h=5,
    timesteps = 1,
)

# # Benchmark Engine: One Cavity
# A system like this with the dissipative Master Equation defined above can be made using a stream of two-level atoms in a Micromaser-like setup [Sec. 5.7 Quantum Collision Models, Ciccarello et al.].
# For example, if maser ancillas are prepared with probability $p$ of beign in the excited state, at resonance, the system will evolve as:
# $$
# \dot{\rho}=(1-p)\Gamma\left(a\rho a^\dagger-\frac{1}{2}\left[a^\dagger a,\rho\right]\right)+p\Gamma\left(a^\dagger\rho a - \frac{1}{2}\left[a a^\dagger,\rho\right]\right)
# $$
# where $\Gamma=g^2\Delta t$ and $g$ is the strength of the cavity-atom interaction.
# 
# The temperature of this "bath" is determined by the population ratio and the energy of the atoms:
# $$
# KT = \frac{\hbar\omega_A}{\log\left(\frac{1-p}{p}\right)}
# $$
# 
# For a **Thermal Engine** we need two thermal baths, and so two gases of atoms with different temperatures given by populations $p_c$ and $p_h$.
class Engine:
    def __init__(self, system, l0=10.0, alpha0=2*np.pi, omega_a=None, omega_dissip=1e-2, v=1e-2, timesteps=1, dt=1e-4, **kwargs):
        self.ndims = system.dims[0][0]
        self.l0 = l0
        self.alpha0 = alpha0
        self.omega_a = omega_a if omega_a else alpha0 / l0
        self.omega_dissip = omega_dissip
        self.v = v
        self.v_t = v
        
        self._system = system
        
        self.t = 0  # Actual time of the machine
        self.dt = dt  # Minimum time difference
        self.timesteps = timesteps  # Temporal length of each stroke
        self.l = l0
        
    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, system):
        if system.dims[0][0] != self.ndims:
            raise ValueError('The system state has a different number of dimensions')
        
        self._system = system
        
    def integrate(self, integrand, ti, tf):
        """Perform an approximated integration in the interval [ti, tf] in discrete time"""
        steps = int((tf - ti) / self.dt)
        integral = 0
        for i in np.linspace(ti, tf, steps):
            integral += integrand(i) * self.dt
        return integral
    
    def differentiate(self, f, t):
        """Approximate the derivative of a function f at time t"""
        return (f(t + self.dt) - f(t)) / self.dt

    def projective_measurement(self, system):
        # SIMPLIFICATION: Take just the diagonal real part of the density matrix
        system = qt.Qobj(np.diag(system.diag().real), dims=system.dims)
        
        return system

    @staticmethod
    def save(filename='engine.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(engine, file)
            
    def save_system_state(self, filename='system'):
        qt.qsave(self.system, filename)
        
    @staticmethod
    def load_system_state(filename='system'):
        system = qt.qload(filename)
        return system
        

        

class MaserEngine(Engine):
    def __init__(self, *args, p_c=0.2, p_h=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_c = p_c
        self.p_h = p_h
        self.states = [(self.energy(), self.alpha0 / self.l)]
                
    def hc(self):
        """Hamiltonian of the cavity"""
        omega_c = self.alpha0 / self.l
        return omega_c * ( qt.create(self.ndims) * qt.destroy(self.ndims) + 1/2)
    
    def hamiltonian(self):
        """In this Hamiltonian the time dependance is implicit in the variable self.l"""
        return self.hc()
    
    def dissipative_meq(self, system, p):
        def commutator(rho, hamiltonian):
            return -1j * (hamiltonian * rho - rho * hamiltonian)
        
        def dissipator(rho, p):
            ap = qt.create(self.ndims)
            am = qt.destroy(self.ndims)
            up_term = ap * rho * am - 1/2 * ( am * ap * rho + rho * am * ap)
            down_term = am * rho * ap - 1/2 * ( ap * am * rho + rho * ap * am)
            return self.omega_dissip * p * up_term + self.omega_dissip * (1 - p) * down_term
        
        return commutator(system, self.hamiltonian()) + dissipator(system, p)
    
    def energy(self):
        return qt.expect(self.hamiltonian(), self.system)
    
    def plot_length_vs_energy(self, start_from=0):
        fig, ax = plt.subplots(figsize=(5, 5))
        y = np.array(self.states)[start_from:, 0] 
        x = np.array(self.states)[start_from:, 1]
        ax.plot(x, y)
        ax.set_xlabel('$\omega$')
        ax.set_ylabel('$E\;\;$', rotation=0)
        plt.show()


class OttoEngine(MaserEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def adiabatic_stroke(self, system):
        steps = int(1 / self.dt)
        
        for i in np.linspace(0, self.timesteps, steps):
            # Move the piston
            self.l = self.l + self.v_t * self.dt
            exp = -1j * self.hamiltonian() * self.dt
            u = exp.expm()
            system = u * system * u.dag()

        self.t += 1
        return self.projective_measurement(system)
    
    def isochoric_stroke(self, system, p):
        """This is a heating stroke with the piston (volume) fixed"""
        steps = int(1 / self.dt)
        # Here the piston is fixed
        for i in np.linspace(0, self.timesteps, steps):
            drdt = self.dissipative_meq(system, p)
            system = system + drdt * self.dt
            
        self.t += 1
        return self.projective_measurement(system)
    
    def cycle(self):
        """One pass of the Otto cycle"""
        # 1) Adiabatic Compression
        self.v_t = -1 * abs(self.v)
        self.system = self.adiabatic_stroke(self.system)
        self.states.append((self.energy(), self.alpha0 / self.l))
        
        # 2) Isochoric Heating
        self.v_t = 0
        self.system = self.isochoric_stroke(self.system, self.p_h)
        self.states.append((self.energy(), self.alpha0 / self.l))
        
        # 3) Adiabatic Expansion
        self.v_t = abs(self.v)
        self.system = self.adiabatic_stroke(self.system)
        self.states.append((self.energy(), self.alpha0 / self.l))
        
        # 4) Isochoric Cooling
        self.v_t = 0
        self.system = self.isochoric_stroke(self.system, self.p_c)
        self.states.append((self.energy(), self.alpha0 / self.l))
        
        return self.system

    
if __name__ == '__main__':
    # ### Setup the Bath
    def boltzmann_population(omega, temp):
        return 1 / (np.exp(omega / temp) + 1)

    starting_len = SIM_PARAMS['l0']
    max_len =  starting_len + SIM_PARAMS['v'] * SIM_PARAMS['timesteps']
    min_len =  starting_len - SIM_PARAMS['v'] * SIM_PARAMS['timesteps']

    omega_c = SIM_PARAMS['alpha0'] / max_len
    omega_h = SIM_PARAMS['alpha0'] / min_len

    p_c = boltzmann_population(omega_c, SIM_PARAMS['T_c'])
    p_h = boltzmann_population(omega_h, SIM_PARAMS['T_h'])
    # system = qt.thermal_dm(NDIMS, 3)
    system = qt.qload('maser_engine_01')

    engine = OttoEngine(system, p_c=p_c, p_h=p_h, **SIM_PARAMS)
    print(engine.energy())
    for c in tqdm.tqdm(range(500)):
        engine.cycle()
        if c % 50 == 0:
            print(engine.energy())

    print(engine.energy())
    engine.plot_length_vs_energy(start_from=-25)
    engine.save_system_state('maser_engine_01')
    with open("..\data\experiment_01.pkl", "wb") as f:
        pickle.dump(SIM_PARAMS, f)
