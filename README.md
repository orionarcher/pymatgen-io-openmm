### Pymatgen-IO-OpenMM


`pymatgen-io-openmm` makes setting up OpenMM simulations fast and easy.

The documentation is under active development. If you would like to use this package,
please contact [Orion Cohen](https://orioncohen.com/) so that he makes
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

### Developer Mode Installation Instructions

1. Install the conda package manager and create a fresh conda environment
with python 3.10 or greater.

    ```bash
    conda create -n openmm python=3.10
    ```

2. Clone this repository and move into it.

    ```bash
    git clone git@github.com:orionarcher/pymatgen-io-openmm.git
    cd pymatgen-io-openmm
    ```

3. Install the requirements in `requirements.txt`.

    ```bash
    conda install -c conda-forge --file requirements.txt
    ```

4. Install the package in editable mode.

    ```bash
    pip install -e .
    ```

### Installing on Apple Silicon

If you are using Apple Silicon, installation is a bit trickier. Some dependencies of OpenFF
are not compatible with Apple Silicon so you will need to use Rosetta 2. First open your 
terminal application with Rosetta2 enabled, then create a conda environment following 
[these instructions](https://docs.openforcefield.org/projects/toolkit/en/stable/installation.html)
everything should work from there. This replaces step 1. Do not use `mamba`, which is not 
fully integrated with x86 installations for Apple Silicon.

### Running tests

To run the testing suite, run following commands from the root of the repository and with your
conda environment activated:

```bash
cd /pymatgen/io/openmm/tests
pytest .
```

Data files paths within the testing suite assuming a path relative to `pymatgen/io/openmm/tests`.
