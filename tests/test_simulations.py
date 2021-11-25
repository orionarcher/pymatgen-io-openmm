# base python
import pytest

# scipy
import numpy as np

# pymatgen
from pymatgen.io.openmm.generators import OpenMMSolutionGen
from pymatgen.io.openmm.simulations import equilibrate_pressure, anneal

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


@pytest.fixture
def ethanol_simulation():
    generator = OpenMMSolutionGen(packmol_random_seed=1)
    input_set = generator.get_input_set({"O": 600, "CCO": 50}, density=1.0)
    return input_set.get_simulation()


def test_equilibrate_pressure(ethanol_simulation):
    ethanol_simulation.minimizeEnergy()
    equilibrate_pressure(ethanol_simulation, 300)
    end_time = ethanol_simulation.context.getState().getTime()._value  # picoseconds
    np.testing.assert_almost_equal(0.300, end_time)


def test_anneal(ethanol_simulation):
    ethanol_simulation.minimizeEnergy()
    anneal(ethanol_simulation, 400, [100, 100, 100], 10)
    end_time = ethanol_simulation.context.getState().getTime()._value  # picoseconds
    np.testing.assert_almost_equal(0.300, end_time)
    end_temp = ethanol_simulation.integrator.getTemperature()._value  # Kelvin
    np.testing.assert_almost_equal(298, end_temp)
