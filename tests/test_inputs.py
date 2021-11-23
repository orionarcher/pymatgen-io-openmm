# base python
import pytest
import tempfile

# cheminformatics
import numpy as np
import parmed

# openff
import openff.toolkit.topology
from openff.toolkit.typing.engines import smirnoff

# openmm
import openmm
from openmm.unit import *
from openmm import NonbondedForce

# pymatgen
import pymatgen
from pymatgen.io.openmm.inputs import (
    TopologyInput,
    SystemInput,
    IntegratorInput,
    StateInput,
    OpenMMSet,
    OpenMMGenerator,
)
from pymatgen.io.openmm.tests.datafiles import (
    input_set_path,
    topology_path,
    state_path,
    corrupted_state_path,
    integrator_path,
    system_path,
    CCO_xyz,
    CCO_charges,
    FEC_r_xyz,
    FEC_s_xyz,
    FEC_charges,
    PF6_xyz,
    PF6_charges,
)


class TestInputFiles:
    def test_topology_input(self):
        topology_input = TopologyInput.from_file(topology_path)
        with tempfile.NamedTemporaryFile() as f:
            topology_input.write_file(f.name)
            topology_input = TopologyInput.from_file(f.name)
        topology = topology_input.get_topology()
        assert isinstance(topology, openmm.app.Topology)
        assert topology.getNumAtoms() == 780
        assert topology.getNumResidues() == 220
        assert topology.getNumBonds() == 560

    def test_system_input(self):
        system_input = SystemInput.from_file(system_path)
        with tempfile.NamedTemporaryFile() as f:
            system_input.write_file(f.name)
            system_input = SystemInput.from_file(f.name)
        system = system_input.get_system()
        assert isinstance(system, openmm.System)
        assert system.getNumParticles() == 780
        assert system.usesPeriodicBoundaryConditions()

    def test_integrator_input(self):
        integrator_input = IntegratorInput.from_file(integrator_path)
        with tempfile.NamedTemporaryFile() as f:
            integrator_input.write_file(f.name)
            integrator_input = IntegratorInput.from_file(f.name)
        integrator = integrator_input.get_integrator()
        assert isinstance(integrator, openmm.Integrator)
        assert integrator.getStepSize() == 1 * femtoseconds
        assert integrator.getTemperature() == 298 * kelvin

    def test_state_input(self):
        state_input = StateInput.from_file(state_path)
        with tempfile.NamedTemporaryFile() as f:
            state_input.write_file(f.name)
            state_input = StateInput.from_file(f.name)
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
        assert len(struct2.atoms) == 3
        assert len(struct2.residues) == 1
        assert len(struct2.bonds) == 2
        struct3 = OpenMMGenerator._smile_to_parmed_structure("O=C1OC[C@H](F)O1")
        assert len(struct3.atoms) == 10
        assert len(struct3.residues) == 1
        assert len(struct3.bonds) == 10

    def test_get_openmm_topology(self):
        topology = OpenMMGenerator._get_openmm_topology({"O": 200, "CCO": 20})
        assert isinstance(topology, openmm.app.Topology)
        assert topology.getNumAtoms() == 780
        assert topology.getNumResidues() == 220
        assert topology.getNumBonds() == 560
        ethanol_smile = "CCO"
        fec_smile = "O=C1OC[C@H](F)O1"
        topology = OpenMMGenerator._get_openmm_topology(
            {ethanol_smile: 50, fec_smile: 50}
        )
        assert topology.getNumAtoms() == 950

    def test_get_box(self):
        box = OpenMMGenerator.get_box({"O": 200, "CCO": 20}, 1)
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
        "xyz_path, n_atoms, n_bonds",
        [
            (CCO_xyz, 9, 8),
            (FEC_r_xyz, 10, 10),
            (FEC_s_xyz, 10, 10),
            (PF6_xyz, 7, 6),
        ],
    )
    def test_infer_openff_mol(self, xyz_path, n_atoms, n_bonds):
        mol = pymatgen.core.Molecule.from_file(xyz_path)
        openff_mol = OpenMMGenerator._infer_openff_mol(mol)
        assert isinstance(openff_mol, openff.toolkit.topology.Molecule)
        assert openff_mol.n_atoms == n_atoms
        assert openff_mol.n_bonds == n_bonds

    @pytest.mark.parametrize(
        "xyz_path, smile, map_values",
        [
            (CCO_xyz, "CCO", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
            (FEC_r_xyz, "O=C1OC[C@@H](F)O1", [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]),
            (FEC_s_xyz, "O=C1OC[C@H](F)O1", [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]),
            (PF6_xyz, "F[P-](F)(F)(F)(F)F", [1, 0, 2, 3, 4, 5, 6]),
        ],
    )
    def test_get_atom_map(self, xyz_path, smile, map_values):
        mol = pymatgen.core.Molecule.from_file(xyz_path)
        inferred_mol = OpenMMGenerator._infer_openff_mol(mol)
        openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
        isomorphic, atom_map = OpenMMGenerator._get_atom_map(inferred_mol, openff_mol)
        assert isomorphic
        assert map_values == list(atom_map.values())

    @pytest.mark.parametrize(
        "charges_path, smile, atom_values",
        [
            (CCO_charges, "CCO", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
            (FEC_charges, "O=C1OC[C@@H](F)O1", [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]),
            (FEC_charges, "O=C1OC[C@H](F)O1", [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]),
            (PF6_charges, "F[P-](F)(F)(F)(F)F", [1, 0, 2, 3, 4, 5, 6]),
        ],
    )
    def test_assign_charges_to_openff_mol(self, charges_path, smile, atom_values):
        charges = np.load(charges_path)
        openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
        atom_map = {i: j for i, j in enumerate(atom_values)}  # this save some space
        new_mol = OpenMMGenerator._assign_charges_to_openff_mol(
            openff_mol, charges, atom_map
        )
        atom_map_inverse = {j: i for i, j in atom_map.items()}
        mapped_charges = [charges[atom_map_inverse[i]] for i in range(len(charges))]
        np.testing.assert_almost_equal(mapped_charges, new_mol.partial_charges._value)

    @pytest.mark.parametrize(
        "charges_path, smile, atom_values",
        [
            (CCO_charges, "CCO", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
            (FEC_charges, "O=C1OC[C@@H](F)O1", [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]),
            (FEC_charges, "O=C1OC[C@H](F)O1", [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]),
            (PF6_charges, "F[P-](F)(F)(F)(F)F", [1, 0, 2, 3, 4, 5, 6]),
        ],
    )
    def test_add_mol_charges_to_forcefield(self, charges_path, smile, atom_values):
        charges = np.load(charges_path)
        openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
        atom_map = {i: j for i, j in enumerate(atom_values)}  # this saves some space
        new_mol = OpenMMGenerator._assign_charges_to_openff_mol(
            openff_mol, charges, atom_map
        )
        atom_map_inverse = {j: i for i, j in atom_map.items()}
        mapped_charges = [charges[atom_map_inverse[i]] for i in range(len(charges))]
        forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
        OpenMMGenerator._add_mol_charges_to_forcefield(forcefield, new_mol)
        topology = openff_mol.to_topology()
        system = forcefield.create_openmm_system(topology)
        for force in system.getForces():
            if type(force) == NonbondedForce:
                for i in range(force.getNumParticles()):
                    assert force.getParticleParameters(i)[0]._value == mapped_charges[i]

    def test_add_partial_charges_to_forcefield(self):
        # set up partial charges
        ethanol_mol = pymatgen.core.Molecule.from_file(CCO_xyz)
        fec_mol = pymatgen.core.Molecule.from_file(FEC_s_xyz)
        ethanol_charges = np.load(CCO_charges)
        fec_charges_og = np.load(FEC_charges)
        partial_charges = [(ethanol_mol, ethanol_charges), (fec_mol, fec_charges_og)]
        # set up force field
        ethanol_smile = "CCO"
        fec_smile = "O=C1OC[C@H](F)O1"
        openff_mols = [
            openff.toolkit.topology.Molecule.from_smiles(smile)
            for smile in [ethanol_smile, fec_smile]
        ]
        openff_forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
        openff_forcefield = OpenMMGenerator._add_partial_charges_to_forcefield(
            openff_forcefield,
            openff_mols,
            partial_charges,
        )
        # construct a System to make testing easier
        topology = OpenMMGenerator._get_openmm_topology(
            {ethanol_smile: 50, fec_smile: 50}
        )
        openff_topology = openff.toolkit.topology.Topology.from_openmm(
            topology, openff_mols
        )
        system = openff_forcefield.create_openmm_system(openff_topology)
        # ensure that all forces are from our assigned force field
        # this does not ensure correct ordering, as we already test that with
        # other methods
        fec_charges = fec_charges_og[[0, 1, 2, 3, 4, 6, 7, 8, 9, 5]]
        full_partial_array = np.append(
            np.tile(ethanol_charges, 50), np.tile(fec_charges, 50)
        )
        for force in system.getForces():
            if type(force) == NonbondedForce:
                charge_array = np.zeros(force.getNumParticles())
                for i in range(len(charge_array)):
                    charge_array[i] = force.getParticleParameters(i)[0]._value
        np.testing.assert_allclose(full_partial_array, charge_array, atol=0.0001)

    def test_parameterize_system(self):
        # TODO: add test here to see if I am adding charges?
        topology = OpenMMGenerator._get_openmm_topology({"O": 200, "CCO": 20})
        smile_strings = ["O", "CCO"]
        box = [0, 0, 0, 19.59, 19.59, 19.59]
        force_field = "Sage"
        system = OpenMMGenerator._parameterize_system(
            topology, smile_strings, box, force_field, []
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
