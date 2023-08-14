# base python
import pytest

# scipy
import numpy as np

# pymatgen
from pymatgen.io.openmm.generators import OpenMMSolutionGen
from pymatgen.io.openmm.sets import OpenMMAlchemySet
from pymatgen.io.openmm.simulations import equilibrate_pressure, anneal, react_system

from datafiles import alchemy_input_set_path

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


@pytest.fixture
def ethanol_simulation():
    generator = OpenMMSolutionGen(packmol_random_seed=1)
    input_mol_dicts = [
        {"smile": "O", "count": 600},
        {"smile": "CCO", "count": 50},
    ]
    input_set = generator.get_input_set(input_mol_dicts, density=1.0)

    return input_set.get_simulation()


def test_equilibrate_pressure(ethanol_simulation):
    ethanol_simulation.minimizeEnergy()
    equilibrate_pressure(ethanol_simulation, 300)
    end_time = ethanol_simulation.context.getState().getTime()._value  # picoseconds
    np.testing.assert_almost_equal(0.300, end_time)


def test_anneal(ethanol_simulation):
    ethanol_simulation.minimizeEnergy()
    anneal(ethanol_simulation, 400, (100, 100, 100), 10)
    end_time = ethanol_simulation.context.getState().getTime()._value  # picoseconds
    np.testing.assert_almost_equal(0.300, end_time)
    end_temp = ethanol_simulation.integrator.getTemperature()._value  # Kelvin
    np.testing.assert_almost_equal(298, end_temp)


def test_react_system():
    # 20 water, 40 ethanol, 40 acetic
    input_set = OpenMMAlchemySet.from_directory(alchemy_input_set_path)
    assert input_set.validate()
    input_set_2 = react_system(
        input_set,
        n_cycles=1,
        steps_per_cycle=200,
    )
    topology = input_set_2.inputs[
        "reactive_system.json"
    ].reactive_system.generate_topology()

    [mol for mol in topology.unique_molecules]
    [mol.n_atoms for mol in topology.molecules]
    # TODO: need to test with delete atoms
    # TODO: should also test with multiple reactions
    return
