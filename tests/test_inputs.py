import pytest
import numpy as np

from pymatgen.io.openmm.inputs import (
    TopologyInput,
    SystemInput,
    IntegratorInput,
    StateInput,
    OpenMMSet,
    OpenMMGenerator,
)

from pymatgen.io.openmm.tests.datafiles import (
    topology_path,
    state_path,
    integrator_path,
    system_path,
    coordinates_path,
)

import pymatgen
import parmed
import openmm


@pytest.fixture
def input_set():
    return


class TestInputFiles:
    def test_debug_topology(self):
        topology = OpenMMGenerator._get_openmm_topology({"O": 200, "CCO": 20})
        positions = np.load(coordinates_path)
        topology_input = TopologyInput(topology)
        print("hi")

    def test_topology_input(self):
        topology_input = TopologyInput.from_file(topology_path)
        return

    def test_system_input(self):
        return

    def test_integrator_input(self):
        integrator = 1
        return

    def test_state_input(self):
        return


class TestOpenMMSet:
    def test_from_directory(self):
        return

    def test_validate(self):
        return

    def test_get_simulation(self):
        return


class TestOpenMMGenerator:
    def test_smile_to_molecule(self):
        mol = OpenMMGenerator._smile_to_molecule("CCO")
        assert isinstance(mol, pymatgen.core.structure.Molecule)
        assert len(mol.sites) == 9

    def test_smile_to_parmed_structure(self):
        struct1 = OpenMMGenerator._smile_to_parmed_structure("CCO")
        assert isinstance(struct1, parmed.Structure)
        assert len(struct1.atoms) == 9
        assert len(struct1.residues) == 1
        assert len(struct1.bonds) == 8
        struct2 = OpenMMGenerator._smile_to_parmed_structure("O")
        # currently failing because H2O is two residues. Issue raised on GitHub
        # https://github.com/openbabel/openbabel/issues/2431
        assert len(struct2.atoms) == 3
        assert len(struct2.residues) == 1
        assert len(struct2.bonds) == 2

    def test_get_openmm_topology(self):
        topology = OpenMMGenerator._get_openmm_topology({"O": 200, "CCO": 20})
        assert isinstance(topology, openmm.app.Topology)
        assert topology.getNumAtoms() == 780
        assert topology.getNumResidues() == 220
        assert topology.getNumBonds() == 560

    def test_get_box(self):
        box = OpenMMGenerator._get_box({"O": 200, "CCO": 20}, 1)
        assert isinstance(box, list)
        assert len(box) == 6
        np.testing.assert_allclose(box[0:3], 0, 2)
        np.testing.assert_allclose(box[3:6], 19.59, 2)

    def test_get_coordinates(self):
        coordinates = OpenMMGenerator._get_coordinates(
            {"O": 200, "CCO": 20}, [0, 0, 0, 19.59, 19.59, 19.59]
        )
        assert isinstance(coordinates, np.ndarray)
        assert len(coordinates) == 780
        assert np.min(coordinates) > -0.2
        assert np.max(coordinates) < 19.8
        assert coordinates.size == 780 * 3

    def test_parameterize_system(self):
        topology = OpenMMGenerator._get_openmm_topology({"O": 200, "CCO": 20})
        smile_strings = ["O", "CCO"]
        box = [0, 0, 0, 19.59, 19.59, 19.59]
        force_field = "Sage"
        system = OpenMMGenerator._parameterize_system(
            topology, smile_strings, box, force_field
        )
        assert system.getNumParticles() == 780
        assert system.usesPeriodicBoundaryConditions()

    def test_get_input_set(self):
        generator = OpenMMGenerator()
        input_set = generator.get_input_set({"O": 200, "CCO": 20}, density=1)
        assert isinstance(input_set, OpenMMSet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
        }
        input_set.get_simulation()
