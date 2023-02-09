import numpy as np
import pytest
import openff.toolkit.topology
import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph

# from openff.toolkit.typing.engines import smirnoff
import openff.toolkit as tk

# openmm
from openmm.unit import elementary_charge

# from openmm.openmm import NonbondedForce

from pymatgen.io.openmm.utils import (
    get_box,
    get_atom_map,
    get_coordinates,
    smiles_to_atom_type_array,
    smiles_to_resname_array,
    xyz_to_molecule,
    molgraph_to_openff_mol,
    openff_mol_to_molgraph,
    molgraph_from_openff_topology,
    get_openff_topology,
    infer_openff_mol,
)

from pymatgen.io.openmm.tests.datafiles import (
    FEC_r_xyz,
    PF6_xyz,
    trimer_txt,
    trimer_pdb,
    CCO_xyz,
    # CCO_charges,
    FEC_s_xyz,
    # FEC_charges,
    # Li_charges,
    PF6_charges,
)


def test_xyz_to_molecule():
    # TODO: add test
    return


def test_smiles_to_atom_type_array():
    smiles = {"O": 5, "CCO": 2}
    atom_type_array = smiles_to_atom_type_array(smiles)
    assert atom_type_array[0] == 0
    assert atom_type_array[15] == 3


def test_smiles_to_resname_array():
    smiles = {"O": 5, "CCO": 2}
    resname_array = smiles_to_resname_array(smiles)
    assert resname_array[0] == "O"
    assert resname_array[15] == "CCO"


def test_get_box():
    O = tk.Molecule.from_smiles("O")
    CCO = tk.Molecule.from_smiles("CCO")
    box = get_box({O: 200, CCO: 20}, 1)
    assert isinstance(box, list)
    assert len(box) == 6
    np.testing.assert_almost_equal(box[0:3], 0, 2)
    np.testing.assert_almost_equal(box[3:6], 19.59, 2)


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


def test_get_coordinates():
    coordinates = get_coordinates({"O": 200, "CCO": 20}, [0, 0, 0, 19.59, 19.59, 19.59], 1, {})
    assert isinstance(coordinates, np.ndarray)
    assert len(coordinates) == 780
    assert np.min(coordinates) > -0.2
    assert np.max(coordinates) < 19.8
    assert coordinates.size == 780 * 3


def test_get_coordinates_added_geometry():
    pf6_geometry = xyz_to_molecule(PF6_xyz)
    coordinates = get_coordinates(
        {"F[P-](F)(F)(F)(F)F": 1},
        [0, 0, 0, 3, 3, 3],
        1,
        smile_geometries={"F[P-](F)(F)(F)(F)F": pf6_geometry},
    )
    assert len(coordinates) == 7
    np.testing.assert_almost_equal(np.linalg.norm(coordinates[1] - coordinates[4]), 1.6, 3)
    with open(trimer_txt) as file:
        trimer_smile = file.read()
    trimer_geometry = xyz_to_molecule(trimer_pdb)
    coordinates = get_coordinates(
        {trimer_smile: 1},
        [0, 0, 0, 20, 20, 20],
        1,
        smile_geometries={trimer_smile: trimer_geometry},
    )
    assert len(coordinates) == 217


def test_get_openff_topology():
    topology = get_openff_topology({"O": 200, "CCO": 20})
    assert topology.n_atoms == 780
    assert topology.n_molecules == 220
    assert topology.n_bonds == 560


# # TODO: refactor to test new parameterize system
# @pytest.mark.parametrize(
#     "charges_path, smile, atom_values",
#     [
#         (Li_charges, "[Li+]", [0]),
#         (CCO_charges, "CCO", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
#         (FEC_charges, "O=C1OC[C@@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 8, 9, 5]),
#         (FEC_charges, "O=C1OC[C@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 8, 9, 5]),
#         (PF6_charges, "F[P-](F)(F)(F)(F)F", [1, 0, 2, 3, 4, 5, 6]),
#     ],
# )
# def test_add_mol_charges_to_forcefield(charges_path, smile, atom_values):
#     charges = np.load(charges_path)
#     openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
#     atom_map = {i: j for i, j in enumerate(atom_values)}  # this saves some space
#     mapped_charges = np.array([charges[atom_map[i]] for i in range(len(charges))])
#     openff_mol.partial_charges = mapped_charges * elementary_charge
#     forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
#     add_mol_charges_to_forcefield(forcefield, openff_mol)
#     topology = openff_mol.to_topology()
#     system = forcefield.create_openmm_system(topology)
#     for force in system.getForces():
#         if type(force) == NonbondedForce:
#             expected = np.array([force.getParticleParameters(i)[0]._value for i in range(force.getNumParticles())])
#             np.testing.assert_allclose(expected, mapped_charges, atol=0.01)


# # TODO refactor to test new parameterize system
# def test_parameterize_system():
#     # TODO: add test here to see if I am adding charges?
#     topology = get_openmm_topology({"O": 200, "CCO": 20})
#     smile_strings = ["O", "CCO"]
#     box = [0, 0, 0, 19.59, 19.59, 19.59]
#     force_field = "Sage"
#     partial_charge_method = "am1bcc"
#     system = parameterize_system(
#         topology,
#         smile_strings,
#         box,
#         force_field=force_field,
#         partial_charge_method=partial_charge_method,
#         partial_charge_scaling={},
#         partial_charges=[],
#     )
#     assert system.getNumParticles() == 780
#     assert system.usesPeriodicBoundaryConditions()
#
#
# # TODO: refactor to test new parameterize system
# @pytest.mark.parametrize("w_ff, sm_ff", [("spce", "gaff"), ("spce", "sage"), ("tip3p", "gaff")])
# def test_parameterize_mixed_forcefield_system(w_ff, sm_ff):
#     # TODO: test with charges
#     # TODO: periodic boundaries assertion
#     # TODO: assert forcefields assigned correctly
#     topology = get_openmm_topology({"O": 200, "CCO": 20})
#     smile_strings = ["O", "CCO"]
#     box = [0, 0, 0, 19.59, 19.59, 19.59]
#     force_field = {"O": w_ff, "CCO": sm_ff}
#     partial_charge_method = "am1bcc"
#     system = parameterize_system(
#         topology,
#         smile_strings,
#         box,
#         force_field=force_field,
#         partial_charge_method=partial_charge_method,
#         partial_charge_scaling={},
#         partial_charges=[],
#     )
#     assert len(system.getForces()) > 0
#     wforce = system.getForces()[0].getBondParameters(0)
#     notwforce = system.getForces()[0].getBondParameters(401)
#     assert wforce != notwforce
#     assert system.getNumParticles() == 780
#     assert system.usesPeriodicBoundaryConditions()
#
#
# # TODO refactor, perhaps disable for now
# @pytest.mark.parametrize("modela, modelb", [("spce", "tip3p"), ("amber14/tip3p.xml", "amber14/tip3pfb.xml")])
# def test_water_models(modela, modelb):
#     topology = get_openmm_topology({"O": 200})
#     smile_strings = ["O"]
#     box = [0, 0, 0, 19.59, 19.59, 19.59]
#     force_field_a = {"O": modela}
#     partial_charge_method = "am1bcc"
#     system_a = parameterize_system(
#         topology,
#         smile_strings,
#         box,
#         force_field=force_field_a,
#         partial_charge_method=partial_charge_method,
#         partial_charge_scaling={},
#         partial_charges=[],
#     )
#     force_field_b = {"O": modelb}
#     partial_charge_method = "am1bcc"
#     system_b = parameterize_system(
#         topology,
#         smile_strings,
#         box,
#         force_field=force_field_b,
#         partial_charge_method=partial_charge_method,
#         partial_charge_scaling={},
#         partial_charges=[],
#     )
#     force_a = system_a.getForces()[0].getBondParameters(0)
#     force_b = system_b.getForces()[0].getBondParameters(0)
#     # assert rOH is different for two different water models
#     assert force_a[2] != force_b[2]


def test_molgraph_to_openff_mol():
    """transform a water MoleculeGraph to a OpenFF water molecule"""
    pf6_mol = pymatgen.core.Molecule.from_file(PF6_xyz)
    pf6_mol.set_charge_and_spin(charge=-1)
    pf6_graph = MoleculeGraph.with_edges(
        pf6_mol,
        {
            (0, 1): {"weight": 1},
            (0, 2): {"weight": 1},
            (0, 3): {"weight": 1},
            (0, 4): {"weight": 1},
            (0, 5): {"weight": 1},
            (0, 6): {"weight": 1},
        },
    )
    pf6_openff = molgraph_to_openff_mol(pf6_graph)
    pf6_graph2 = openff_mol_to_molgraph(pf6_openff)
    pf6_openff2 = molgraph_to_openff_mol(pf6_graph2)
    assert pf6_openff == pf6_openff2


def test_openff_mol_to_molgraph():
    import networkx as nx
    import networkx.algorithms.isomorphism as iso

    pf6_openff = openff.toolkit.topology.Molecule.from_smiles("F[P-](F)(F)(F)(F)F")
    pf6_charges = np.load(PF6_charges)[[1, 0, 2, 3, 4, 5, 6]]
    pf6_openff.partial_charges = pf6_charges * elementary_charge
    pf6_graph = openff_mol_to_molgraph(pf6_openff)
    assert len(pf6_graph.molecule) == 7
    assert pf6_graph.molecule.charge == -1
    em = iso.categorical_edge_match("weight", 1)
    nm = iso.numerical_node_match(["formal_charge", "partial_charge"], [0, 0])
    pf6_openff2 = molgraph_to_openff_mol(pf6_graph)
    pf6_graph2 = openff_mol_to_molgraph(pf6_openff2)
    assert nx.is_isomorphic(pf6_graph.graph, pf6_graph2.graph, edge_match=em, node_match=nm)
    assert pf6_graph.molecule == pf6_graph2.molecule
    assert pf6_openff == pf6_openff2


def test_molgraph_from_openff_topology():
    topology = get_openff_topology({"O": 200, "CCO": 20})
    molgraph_from_openff_topology(topology)
