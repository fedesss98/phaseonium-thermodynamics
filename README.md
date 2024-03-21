# Fully-Coherent-Phaseonium Aided Thermodynamic Cycle

This repository contains a Python project that simulates the dynamics of a cavity in contact with three-level phaseonium atoms, using the collision model framework. The cavity is the System and the phaseonium atoms are the Ancillas.
The dynamical map that describes the evolution of the System-Ancilla state was found by Amato, F. and Pellitteri, C. 

## Features

- Create a Cavity system with specific parameters (thermal, coherent...).
- Create an Ancilla instance with specific parameters for the internal coherences.
- Set and get the attributes of an Ancilla instance.
- Save the parameters of an Ancilla instance to a JSON file.
- Run the evolution of the system under the interaction with always new Ancillas.
- Track the evolution of the average Photon Number in the Cavity and its Temperature (if that is defined).
- Make a .gif image of the evolution of the Wigner function of the system.

## Usage

### Creating an Ancilla Instance

```python
ancilla_instance = Ancilla(qt.qeye(3))  # qt.qeye(3) creates a 3x3 identity matrix
```

### Setting an Attribute
Phaseonium parameters can be changed dynamically
```python
ancilla_instance.chi01 = complex(1, 2)  # You can replace complex(1, 2) with the desired complex number
```

### Saving Ancilla Parameters

```python
save_id = ancilla_instance.save_parameters("unique_id")  # Replace "unique_id" with a unique identifier
```

## Dependencies

This project requires the following Python libraries:

- QuTiP
- NumPy
- json
- os

## Installation

To install the required libraries, run the following command:

```bash
pip install qutip numpy
```

## Contributing

Contributions are welcome. Please submit a pull request or create an issue to discuss the changes you want to make.

## License

This project is licensed under the MIT License.