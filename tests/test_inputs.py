import pathlib
import tempfile

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
    test_files_path,
    input_set_path,
    topology_path,
    state_path,
    corrupted_state_path,
    integrator_path,
    system_path,
    coordinates_path,
)

import pymatgen
import parmed
import openmm
from openmm.unit import *


@pytest.fixture
def input_set():
    return


class TestInputFiles:
    def test_topology_input(self):
        topology_input = TopologyInput.from_file(topology_path)
        with tempfile.NamedTemporaryFile() as f:
            topology_input.write_file(f.name)
        topology = topology_input.get_topology()
        assert isinstance(topology, openmm.app.Topology)
        assert topology.getNumAtoms() == 780
        assert topology.getNumResidues() == 220
        assert topology.getNumBonds() == 560

    def test_system_input(self):
        system_input = SystemInput.from_file(system_path)
        with tempfile.NamedTemporaryFile() as f:
            system_input.write_file(f.name)
        system = system_input.get_system()
        assert isinstance(system, openmm.System)
        assert system.getNumParticles() == 780
        assert system.usesPeriodicBoundaryConditions()

    def test_integrator_input(self):
        integrator_input = IntegratorInput.from_file(integrator_path)
        with tempfile.NamedTemporaryFile() as f:
            integrator_input.write_file(f.name)
        integrator = integrator_input.get_integrator()
        assert isinstance(integrator, openmm.Integrator)
        assert integrator.getStepSize() == 1 * femtoseconds
        assert integrator.getTemperature() == 298 * kelvin

    def test_state_input(self):
        state_input = StateInput.from_file(state_path)
        with tempfile.NamedTemporaryFile() as f:
            state_input.write_file(f.name)
        state = state_input.get_state()
        assert isinstance(state, openmm.State)
        assert len(state.getPositions(asNumpy=True)) == 780


class TestOpenMMSet:
    def test_from_directory(self):
        input_set = OpenMMSet.from_directory(input_set_path)
        assert len(input_set.inputs) == 4
        assert input_set.topology_file == "topology.pdb"
        assert input_set.state_file == "state.xml"
        assert isinstance(input_set.inputs["topology.pdb"], TopologyInput)
        assert isinstance(input_set.inputs["state.xml"], StateInput)
        input_set2 = OpenMMSet.from_directory(
            input_set_path, state_file="wrong_file.xml"
        )
        assert len(input_set2.inputs) == 3
        assert input_set2.topology_file == "topology.pdb"
        assert input_set2.state_file is None

    def test_validate(self):
        input_set = OpenMMSet.from_directory(input_set_path)
        assert input_set.validate()
        corrupted_input_set = OpenMMSet.from_directory(
            input_set_path, state_file=corrupted_state_path
        )
        assert not corrupted_input_set.validate()

    def test_get_simulation(self):
        input_set = OpenMMSet.from_directory(input_set_path)
        simulation = input_set.get_simulation()
        state = simulation.context.getState(getPositions=True)
        assert len(state.getPositions(asNumpy=True)) == 780
        assert simulation.system.getNumParticles() == 780
        assert simulation.system.usesPeriodicBoundaryConditions()
        assert simulation.topology.getNumAtoms() == 780
        assert simulation.topology.getNumResidues() == 220
        assert simulation.topology.getNumBonds() == 560


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

    @pytest.mark.parametrize(
        "mol_name",
        ["FEC", "CCO", "PF6"],
    )
    def test_get_charged_openff_mol(self, mol_name):

        return

    def test_add_mole_charges_to_forcefield(self):
        return

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
        assert input_set.validate()
