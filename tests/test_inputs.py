import pytest

from pymatgen.io.openmm.inputs import (
    TopologyInput,
    SystemInput,
    IntegratorInput,
    StateInput,
    OpenMMSet,
    OpenMMGenerator,
)

import pymatgen
import parmed
import openmm

class TestInputFiles:
    def test_topology_input(self):
        return

    def test_system_input(self):
        return

    def test_integrator_input(self):
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
        assert len(struct1.residues) == 2
        assert len(struct1.bonds) == 2

    def test_get_openmm_topology(self):
        topology = OpenMMGenerator._get_openmm_topology({"O": 200, "CCO": 20})
        assert isinstance(topology, openmm.app.Topology)
        assert topology.getNumAtoms() == 780
        assert topology.getNumResidues() == 220
        assert topology.getNumBonds() == 560
        return

    def test_get_box(self):
        return

    def test_get_coordinates(self):
        return

    def test_parameterize_system(self):
        return

    def test_get_input_set(self):
        return
