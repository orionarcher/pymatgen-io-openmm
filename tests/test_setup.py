import numpy as np
import openmm.app
import pytest

import parmed
from pymatgen.core.structure import Molecule

from pymatgen.io.openmm.setup import (
    OpenMMSimulationGenerator,
    smile_to_mol,
)

@pytest.fixture
def ethanol_water():
    return {"O": 1000, "CCO": 100}


@pytest.fixture
def simulation_generator(ethanol_water):
    generator = OpenMMSimulationGenerator(
        ethanol_water,
        0.98,
    )
    return generator

def test_smile_to_mol():
    mol = smile_to_mol("CCO")
    assert isinstance(mol, Molecule)




class TestOpenMMSimulationGenerator:

    def test_initialization(self, ethanol_water):
        generator = OpenMMSimulationGenerator(
            ethanol_water,
            0.98,
            integrator="LangevinMiddleIntegrator",
            platform="CPU",
        )

    def test_smile_to_parmed_structure(self, simulation_generator, ethanol_water):
        structure = simulation_generator._smile_to_parmed_structure(ethanol_water)
        assert isinstance(structure, parmed.Structure)

    def test_smiles_to_openmm_topology(self, simulation_generator, ethanol_water):
        topology = simulation_generator._smiles_to_openmm_topology(ethanol_water)
        assert isinstance(topology, openmm.app.Topology)

    def test_smiles_to_coordinates(self, simulation_generator, ethanol_water):
        coordinates = simulation_generator._smiles_to_coordinates(ethanol_water, 20)
        assert isinstance(coordinates, np.ndarray)

    def test_smiles_to_cube_size(self, simulation_generator, ethanol_water):
        coordinates = simulation_generator._smile_to_openmm_topology(ethanol_water, 0.98)
        assert isinstance(coordinates, np.ndarray)


    def test_smiles_to_system_topology(self, simulation_generator, ethanol_water):
        system, topology = simulation_generator._smile_to_system_and_topology(ethanol_water)
        assert isinstance(system, openmm.System)
        assert isinstance(topology, openmm.app.Topology)


    def test_return_simulation(self, simulation_generator):
        simulation = simulation_generator.return_simulation
        assert isinstance(simulation, openmm.app.Simulation)
