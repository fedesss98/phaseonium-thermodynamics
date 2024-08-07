import matplotlib.pyplot as plt
import numpy as np
# import cupy as cp
import qutip as qt
from qutip.visualization import plot_fock_distribution, plot_wigner
import os
import json

try:
    import observables as obs
except ModuleNotFoundError:
    from thermophaseonium.utilities import observables as obs


def density_matrix(rho, hermitian=True, normalized=True, threshold=1e-6):
    """
    A proper density matrix must be Hermitian and have trace equal to 1.
    This function checks for that.
    The code allows for non-hermitian or non-normalized matrices, but it is not recommended.
    """
    hermitianity = rho.isherm if hermitian else True
    normality = np.sum(rho.diag()) - 1 < threshold if normalized else True
    return hermitianity and normality


class Ancilla(qt.Qobj):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        if not density_matrix(self):
            raise ValueError("Input is not a valid density matrix")
        self._alpha = np.sqrt(self[0, 0])
        self._beta = np.sqrt(2 * self[1, 1])
        self._chi01 = self[0, 1]
        self._chi02 = self[0, 2]
        self._chi12 = self[1, 2]

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def chi01(self):
        return self._chi01

    @chi01.setter
    def chi01(self, value):
        if isinstance(value, complex):
            pass
        elif isinstance(value, float):
            value = np.cos(value) + 1j * np.sin(value)
        self.data[0, 1] = value
        self.data[1, 0] = value.conjugate()
        self._chi01 = value

    @property
    def chi02(self):
        return self._chi02

    @chi02.setter
    def chi02(self, value):
        if isinstance(value, complex):
            pass
        elif isinstance(value, float):
            value = np.cos(value) + 1j * np.sin(value)
        self.data[0, 2] = value
        self.data[2, 0] = value.conjugate()
        self._chi02 = value

    @property
    def chi12(self):
        return self._chi12

    @chi12.setter
    def chi12(self, value):
        if isinstance(value, complex):
            pass
        elif isinstance(value, float):
            value = np.cos(value) + 1j * np.sin(value)
        self.data[1, 2] = value
        self.data[2, 1] = value.conjugate()
        self._chi12 = value

    @property
    def ga(self):
        return np.real(2 * self._alpha ** 2)

    @property
    def gb(self):
        return np.real(self._beta ** 2 + self._chi12 + self._chi12.conjugate())

    @property
    def stable_temperature(self):
        return - 1 / np.log(self.ga / self.gb)

    def save_parameters(self, save_id):
        parameters = {
            "alpha": str(self.alpha),
            "beta": str(self.beta),
            "chi01": str(self.chi01),
            "chi02": str(self.chi02),
            "chi12": str(self.chi12),
        }
        if os.path.exists("ancillas.json"):
            with open("ancillas.json", "r") as f:
                data = json.load(f)
        else:
            data = {}

        # Check if there is already an entry with the same parameters
        for existing_id, existing_parameters in data.items():
            if existing_parameters == parameters:
                print(f"Ancilla with the same parameters already exists with ID {existing_id}")
                self.json_id = existing_id
                return existing_id
        # Check if ID already exists
        if save_id in data:
            raise ValueError(f"ID {save_id} already exists")
        # Add the new parameters and save the file
        data[save_id] = parameters
        with open("ancillas.json", "w") as f:
            json.dump(data, f)

        print(f"Ancilla saved with ID {save_id}")
        self.json_id = save_id
        return save_id


class Cavity(qt.Qobj):
    def __init__(self, *args, **kwargs):
        self.omega = kwargs.pop('omega', 1)
        hermitian = kwargs.pop('hermitian', True)
        normalized = kwargs.pop('normalized', True)
        super().__init__(*args, **kwargs)
        if not density_matrix(self, hermitian, normalized):
            raise ValueError("Input is not a valid density matrix")
        self.a = qt.destroy(self.shape[0])
        self.ad = self.a.dag()

    @property
    def n(self):
        """Mean photon number of the system"""
        operator = self * self.ad * self.a
        return operator.tr()

    @property
    def temperature(self):
        return obs.temperature(self)

    @property
    def entropy(self):
        return obs.entropy(self)

    # def to_cupy(self):
    #     return cp.asarray(self.full())

    def is_diagonal(self):
        return qt.qdiags(self.diag(), 0) == self

    def plot_fock_distribution(self, show=True, path=None, title=False, **kwargs):
        plot_fock_distribution(self, **kwargs)
        if kwargs.get('ax') is None:
            plt.title(title)
            plt.legend([f"<n> = {self.n:.4f}"])
        else:
            kwargs.get('ax').set_title(title)
            kwargs.get('ax').legend([f"<n> = {self.n:.4f}"])
        if show and kwargs.get('fig') is None and kwargs.get('ax') is None:
            plt.show()
        if path is not None:
            plt.savefig(path)
            plt.close()

    def plot_wigner(self, show=True, path=None, **kwargs):
        title, xlim, ylim = kwargs.pop('title', None), kwargs.pop('xlim', None), kwargs.pop('ylim', None)

        # QuTiP's plot_wigner function
        fig, ax = plot_wigner(self, **kwargs)

        ax.set_title(title)

        if show and kwargs.get('fig') is None and kwargs.get('ax') is None:
            plt.show()
        if path is not None:
            fig.savefig(path)
            plt.close()

        return fig, ax