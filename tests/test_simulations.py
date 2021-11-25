import pytest

from pymatgen.io.openmm.generators import OpenMMSolutionGen
from pymatgen.io.openmm.simulations import equilibrate_pressure, anneal

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


@pytest.fixture
def ethanol_simulation():
    generator = OpenMMSolutionGen()
    input_set = generator.get_input_set({"O": 600, "CCO": 50}, density=1.0)
    return input_set.get_simulation()


def test_anneal(ethanol_simulation):
    ethanol_simulation.minimizeEnergy()
    print(ethanol_simulation.context.getState().getTime())
    anneal(ethanol_simulation, 310, [100, 100, 100], 10)
    print(ethanol_simulation.context.getState().getTime())


def test_equilibrate_pressure(ethanol_simulation):
    ethanol_simulation.minimizeEnergy()
    print(ethanol_simulation.context.getState().getTime())
    equilibrate_pressure(ethanol_simulation, 300)
    print(ethanol_simulation.context.getState().getTime())
