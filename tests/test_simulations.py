import pytest

from pymatgen.io.openmm.setup import OpenMMSimulationGenerator
from pymatgen.io.openmm.simulations import equilibrate_pressure, anneal


@pytest.fixture
def ethanol_simulation():
    generator = OpenMMSimulationGenerator(
        {"O": 600, "CCO": 50},
        0.9,
    )
    return generator.return_simulation()


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
