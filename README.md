### Pymatgen-IO-OpenMM


`pymatgen-io-openmm` makes setting up OpenMM simulations fast and easy.

The documentation is under active development. If you would like to use this package,
please contact the [Orion Cohen](https://orioncohen.com/) so that he makes
time to finish documenting and mint a PyPI release.

`pymatgen-io-openmm` is a [pymatgen](https://pymatgen.org/) namespace package.
You can install this package from source and all functionality
will be available in the pymatgen namespace. A basic usage
example is shown below.

```python
from pymatgen.io.openmm.generators import OpenMMGenerator

# Create a generator object
generator = OpenMMGenerator()

input_mol_dicts = [
    {"smile": "O", "count": 200},
    {"smile": "CCO", "count": 20},
]

input_set = generator.get_input_set(input_mol_dicts, density=1)

simulation = input_set.get_simulation()
simulation.minimizeEnergy()
simulation.step(1000)
```
