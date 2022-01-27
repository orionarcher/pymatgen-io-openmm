import numpy as np
import parmed
import pytest
import openff.toolkit.topology
import pymatgen
import openmm
from openff.toolkit.typing.engines import smirnoff

# openmm
from openmm.unit import elementary_charge
from openmm import NonbondedForce

from pymatgen.io.openmm.utils import (
    get_box,
    smile_to_molecule,
    smile_to_parmed_structure,
    n_mols_from_mass_ratio,
    n_mols_from_volume_ratio,
    n_solute_from_molarity,
    calculate_molarity,
    get_atom_map,
    infer_openff_mol,
    order_molecule_like_smile,
    get_coordinates,
    get_openmm_topology,
    add_mol_charges_to_forcefield,
    assign_charges_to_mols,
)

from pymatgen.io.openmm.tests.datafiles import (
    FEC_r_xyz,
    PF6_xyz,
    trimer_txt,
    trimer_pdb,
    CCO_xyz,
    CCO_charges,
    FEC_s_xyz,
    FEC_charges,
    Li_charges,
    PF6_charges,
)


def test_get_box():
    box = get_box({"O": 200, "CCO": 20}, 1)
    assert isinstance(box, list)
    assert len(box) == 6
    np.testing.assert_almost_equal(box[0:3], 0, 2)
    np.testing.assert_almost_equal(box[3:6], 19.59, 2)


def test_smile_to_parmed_structure():
    struct1 = smile_to_parmed_structure("CCO")
    assert isinstance(struct1, parmed.Structure)
    assert len(struct1.atoms) == 9
    assert len(struct1.residues) == 1
    assert len(struct1.bonds) == 8
    struct2 = smile_to_parmed_structure("O")
    assert len(struct2.atoms) == 3
    assert len(struct2.residues) == 1
    assert len(struct2.bonds) == 2
    struct3 = smile_to_parmed_structure("O=C1OC[C@H](F)O1")
    assert len(struct3.atoms) == 10
    assert len(struct3.residues) == 1
    assert len(struct3.bonds) == 10


def test_smile_to_molecule():
    mol = smile_to_molecule("CCO")
    assert isinstance(mol, pymatgen.core.structure.Molecule)
    assert len(mol.sites) == 9


def test_counts_from_mass_ratio():
    n_mols = n_mols_from_mass_ratio(600, ["O", "CCO"], [10, 1])
    np.testing.assert_allclose(n_mols, [577, 23])


def test_n_mols_from_volume_ratio():
    n_mols = n_mols_from_volume_ratio(600, ["O", "CCO"], [10, 1], [1, 0.79])
    np.testing.assert_allclose(n_mols, [582, 18])


def test_n_solute_from_molarity():
    nm3_to_L = 1e-24
    n_solute = n_solute_from_molarity(1, 2 ** 3 * nm3_to_L)
    np.testing.assert_allclose(n_solute, 5)
    n_solute = n_solute_from_molarity(1, 4 ** 3 * nm3_to_L)
    np.testing.assert_allclose(n_solute, 39)


def test_calculate_molarity():
    nm3_to_L = 1e-24
    np.testing.assert_almost_equal(calculate_molarity(4 ** 3 * nm3_to_L, 39), 1, decimal=1)


@pytest.mark.parametrize(
    "xyz_path, smile, map_values",
    [
        (CCO_xyz, "CCO", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        (FEC_r_xyz, "O=C1OC[C@@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 9, 8, 5]),
        (FEC_s_xyz, "O=C1OC[C@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 9, 8, 5]),
        (PF6_xyz, "F[P-](F)(F)(F)(F)F", [1, 0, 2, 3, 4, 5, 6]),
    ],
)
def test_get_atom_map(xyz_path, smile, map_values):
    mol = pymatgen.core.Molecule.from_file(xyz_path)
    inferred_mol = infer_openff_mol(mol)
    openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
    isomorphic, atom_map = get_atom_map(inferred_mol, openff_mol)
    assert isomorphic
    assert map_values == list(atom_map.values())


@pytest.mark.parametrize(
    "xyz_path, n_atoms, n_bonds",
    [
        (CCO_xyz, 9, 8),
        (FEC_r_xyz, 10, 10),
        (FEC_s_xyz, 10, 10),
        (PF6_xyz, 7, 6),
    ],
)
def test_infer_openff_mol(xyz_path, n_atoms, n_bonds):
    mol = pymatgen.core.Molecule.from_file(xyz_path)
    openff_mol = infer_openff_mol(mol)
    assert isinstance(openff_mol, openff.toolkit.topology.Molecule)
    assert openff_mol.n_atoms == n_atoms
    assert openff_mol.n_bonds == n_bonds


@pytest.mark.parametrize(
    "xyz_path, smile, atomic_numbers",
    [
        (CCO_xyz, "CCO", (6, 6, 8, 1, 1, 1, 1, 1, 1)),
        (PF6_xyz, "F[P-](F)(F)(F)(F)F", (9, 15, 9, 9, 9, 9, 9)),
    ],
)
def test_order_molecule_like_smile(xyz_path, smile, atomic_numbers):
    mol = pymatgen.core.Molecule.from_file(xyz_path)
    ordered_mol = order_molecule_like_smile(smile, mol)
    np.testing.assert_almost_equal(ordered_mol.atomic_numbers, atomic_numbers)


def test_get_coordinates():
    coordinates = get_coordinates({"O": 200, "CCO": 20}, [0, 0, 0, 19.59, 19.59, 19.59], 1, {})
    assert isinstance(coordinates, np.ndarray)
    assert len(coordinates) == 780
    assert np.min(coordinates) > -0.2
    assert np.max(coordinates) < 19.8
    assert coordinates.size == 780 * 3


def test_get_coordinates_added_geometry():
    coordinates = get_coordinates(
        {"F[P-](F)(F)(F)(F)F": 1}, [0, 0, 0, 3, 3, 3], 1, smile_geometries={"F[P-](F)(F)(F)(F)F": PF6_xyz}
    )
    assert len(coordinates) == 7
    np.testing.assert_almost_equal(np.linalg.norm(coordinates[1] - coordinates[4]), 1.6)
    with open(trimer_txt) as file:
        trimer_smile = file.read()
    coordinates = get_coordinates(
        {trimer_smile: 1}, [0, 0, 0, 20, 20, 20], 1, smile_geometries={trimer_smile: trimer_pdb}
    )
    assert len(coordinates) == 217


def test_get_openmm_topology():
    topology = get_openmm_topology({"O": 200, "CCO": 20})
    assert isinstance(topology, openmm.app.Topology)
    assert topology.getNumAtoms() == 780
    assert topology.getNumResidues() == 220
    assert topology.getNumBonds() == 560
    ethanol_smile = "CCO"
    fec_smile = "O=C1OC[C@H](F)O1"
    topology = get_openmm_topology({ethanol_smile: 50, fec_smile: 50})
    assert topology.getNumAtoms() == 950


@pytest.mark.parametrize(
    "charges_path, smile, atom_values",
    [
        (Li_charges, "[Li+]", [0]),
        (CCO_charges, "CCO", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        (FEC_charges, "O=C1OC[C@@H](F)O1", [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]),
        (FEC_charges, "O=C1OC[C@H](F)O1", [0, 1, 2, 3, 4, 9, 5, 6, 7, 8]),
        (PF6_charges, "F[P-](F)(F)(F)(F)F", [1, 0, 2, 3, 4, 5, 6]),
    ],
)
def test_add_mol_charges_to_forcefield(charges_path, smile, atom_values):
    charges = np.load(charges_path)
    openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
    atom_map = {i: j for i, j in enumerate(atom_values)}  # this saves some space
    mapped_charges = np.array([charges[atom_map[i]] for i in range(len(charges))])
    openff_mol.partial_charges = mapped_charges * elementary_charge
    forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
    add_mol_charges_to_forcefield(forcefield, openff_mol)
    topology = openff_mol.to_topology()
    system = forcefield.create_openmm_system(topology)
    for force in system.getForces():
        if type(force) == NonbondedForce:
            for i in range(force.getNumParticles()):
                assert force.getParticleParameters(i)[0]._value == mapped_charges[i]


def test_assign_charges_to_mols():
    # set up partial charges
    ethanol_mol = pymatgen.core.Molecule.from_file(CCO_xyz)
    fec_mol = pymatgen.core.Molecule.from_file(FEC_s_xyz)
    ethanol_charges = np.load(CCO_charges)
    fec_charges = np.load(FEC_charges)
    partial_charges = [(ethanol_mol, ethanol_charges), (fec_mol, fec_charges)]
    # set up force field
    ethanol_smile = "CCO"
    fec_smile = "O=C1OC[C@H](F)O1"
    openff_forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
    charged_mols = assign_charges_to_mols(
        [ethanol_smile, fec_smile],
        "am1bcc",
        {},
        partial_charges,
    )
    openff_forcefield_scaled = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
    charged_mols_scaled = assign_charges_to_mols(
        [ethanol_smile, fec_smile],
        "am1bcc",
        {ethanol_smile: 0.9, fec_smile: 0.9},
        partial_charges,
    )
    # construct a System to make testing easier
    topology = get_openmm_topology({ethanol_smile: 50, fec_smile: 50})
    openff_topology = openff.toolkit.topology.Topology.from_openmm(topology, charged_mols)
    openff_topology_scaled = openff.toolkit.topology.Topology.from_openmm(topology, charged_mols_scaled)
    system = openff_forcefield.create_openmm_system(
        openff_topology,
        charge_from_molecules=charged_mols,
    )
    system_scaled = openff_forcefield_scaled.create_openmm_system(
        openff_topology_scaled,
        charge_from_molecules=charged_mols_scaled,
    )
    # ensure that all forces are from our assigned force field
    # this does not ensure correct ordering, as we already test that with
    # other methods
    fec_charges_reordered = fec_charges[[0, 1, 2, 3, 4, 6, 7, 8, 9, 5]]
    full_partial_array = np.append(np.tile(ethanol_charges, 50), np.tile(fec_charges_reordered, 50))
    for force in system.getForces():
        if type(force) == NonbondedForce:
            charge_array = np.zeros(force.getNumParticles())
            for i in range(len(charge_array)):
                charge_array[i] = force.getParticleParameters(i)[0]._value
    np.testing.assert_allclose(full_partial_array, charge_array, atol=0.0001)
    for force in system_scaled.getForces():
        if type(force) == NonbondedForce:
            charge_array = np.zeros(force.getNumParticles())
            for i in range(len(charge_array)):
                charge_array[i] = force.getParticleParameters(i)[0]._value
    np.testing.assert_allclose(full_partial_array * 0.9, charge_array, atol=0.0001)
