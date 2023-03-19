import numpy as np
import pytest
import openff.toolkit.topology
from monty.json import MontyEncoder
import json

import pymatgen
from pymatgen.analysis.graphs import MoleculeGraph

# from openff.toolkit.typing.engines import smirnoff
import openff.toolkit as tk

# openmm

# from openmm.openmm import NonbondedForce

from pymatgen.io.openmm.utils import (
    get_box,
    get_atom_map,
    get_coordinates,
    smiles_to_atom_type_array,
    smiles_to_resname_array,
    molgraph_to_openff_mol,
    molgraph_from_openff_mol,
    molgraph_from_openff_topology,
    get_openff_topology,
    infer_openff_mol,
    parameterize_w_interchange,
    get_unique_subgraphs,
    molgraph_from_molecules,
    molgraph_to_openff_topology,
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
    # PF6_charges,
    # alchemy_input_set_path,
)
from openff.units import unit

import networkx as nx
import networkx.algorithms.isomorphism as iso


def test_xyz_to_molecule():
    # TODO: add test
    return


def test_smiles_to_atom_type_array():
    O = tk.Molecule.from_smiles("O")
    CCO = tk.Molecule.from_smiles("CCO")
    mols = {O: 5, CCO: 2}
    atom_type_array = smiles_to_atom_type_array(mols)
    assert atom_type_array[0] == 0
    assert atom_type_array[15] == 3


def test_smiles_to_resname_array():
    O = tk.Molecule.from_smiles("O")
    CCO = tk.Molecule.from_smiles("CCO")
    mol_specs = [
        {"openff_mol": O, "count": 5, "name": "O"},
        {"openff_mol": CCO, "count": 5, "name": "CCO"},
    ]
    resname_array = smiles_to_resname_array(mol_specs)
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
    O = tk.Molecule.from_smiles("O")
    CCO = tk.Molecule.from_smiles("CCO")
    O.generate_conformers()
    coords = pymatgen.core.Molecule.from_file(CCO_xyz).cart_coords * unit.angstrom
    CCO.add_conformer(coords)
    CCO.add_conformer(coords)
    CCO.add_conformer(coords)
    coordinates = get_coordinates(
        {O: 200, CCO: 20}, box=[0, 0, 0, 19.59, 19.59, 19.59], random_seed=1
    )
    assert isinstance(coordinates, np.ndarray)
    assert len(coordinates) == 780
    assert np.min(coordinates) > -0.2
    assert np.max(coordinates) < 19.8
    assert coordinates.size == 780 * 3


def test_get_coordinates_added_geometry():
    pf6_mol = pymatgen.core.Molecule.from_file(PF6_xyz)
    pf6 = infer_openff_mol(pf6_mol)
    pf6_geometry = pymatgen.core.Molecule.from_file(PF6_xyz).cart_coords * unit.angstrom
    # pf6 = tk.Molecule.from_smiles("F[P-](F)(F)(F)(F)F")
    pf6.add_conformer(pf6_geometry)
    coordinates = get_coordinates(
        {pf6: 3},
        [0, 0, 0, 3, 3, 3],
        1,
    )
    assert len(coordinates) == 21
    np.testing.assert_almost_equal(
        np.linalg.norm(coordinates[0] - coordinates[4]), 1.6, 3
    )
    with open(trimer_txt) as file:
        trimer_smile = file.read()
    trimer_geometry = (
        pymatgen.core.Molecule.from_file(trimer_pdb).cart_coords * unit.angstrom
    )
    trimer = tk.Molecule.from_smiles(trimer_smile)
    trimer.add_conformer(trimer_geometry)

    coordinates = get_coordinates(
        {trimer: 1},
        [0, 0, 0, 20, 20, 20],
        1,
    )
    assert len(coordinates) == 217


def test_get_openff_topology():
    O = tk.Molecule.from_smiles("O")
    CCO = tk.Molecule.from_smiles("CCO")
    topology = get_openff_topology({O: 200, CCO: 20})
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


def test_parameterize_w_interchange():
    O = tk.Molecule.from_smiles("O")
    CCO = tk.Molecule.from_smiles("CCO")
    O.assign_partial_charges("am1bcc")
    CCO.assign_partial_charges("am1bcc")

    o_spec = dict(
        smile="O",
        count=200,
        name="O",
        forcefield="sage",
        formal_charge=0,
        openff_mol=O,
    )
    cco_spec = dict(
        smile="CCO",
        count=200,
        name="CCO",
        forcefield="sage",
        formal_charge=0,
        openff_mol=CCO,
    )

    topology = get_openff_topology({O: 200, CCO: 20})
    box = np.array([0, 0, 0, 19.59, 19.59, 19.59])
    parameterize_w_interchange(topology, [o_spec, cco_spec], box)


def test_get_unique_subgraphs_basic():
    O = tk.Molecule.from_smiles("O")
    CCO = tk.Molecule.from_smiles("CCO")
    tk.Molecule.from_smiles("CO")

    o_graph = molgraph_from_openff_mol(O)
    cco_graph = molgraph_from_openff_mol(CCO)

    unique_subgraphs = get_unique_subgraphs([o_graph, cco_graph, cco_graph, cco_graph])

    assert len(unique_subgraphs) == 2


def test_get_unique_subgraphs_from_topology():
    O = tk.Molecule.from_smiles("O")
    CCO = tk.Molecule.from_smiles("CCO")
    CO = tk.Molecule.from_smiles("CO")
    topology = get_openff_topology({O: 200, CCO: 20, CO: 20})

    molgraph = molgraph_from_openff_topology(topology)
    subgraphs = molgraph.get_disconnected_fragments()
    unique_subgraphs = get_unique_subgraphs(subgraphs)

    assert len(unique_subgraphs) == 3

    o_graph = molgraph_from_openff_mol(O)
    cco_graph = molgraph_from_openff_mol(CCO)

    unique_subgraphs = get_unique_subgraphs([o_graph, cco_graph, cco_graph, cco_graph])

    assert len(unique_subgraphs) == 2


def test_molgraph_from_atom_bonds():
    import networkx as nx
    import networkx.algorithms.isomorphism as iso

    pf6_openff = openff.toolkit.topology.Molecule.from_smiles("F[P-](F)(F)(F)(F)F")

    pf6_graph = molgraph_from_molecules([pf6_openff])

    assert len(pf6_graph.molecule) == 7
    assert pf6_graph.molecule.charge == -1

    em = iso.categorical_edge_match("weight", 1)

    pf6_openff2 = molgraph_to_openff_mol(pf6_graph)
    pf6_graph2 = molgraph_from_openff_mol(pf6_openff2)
    assert nx.is_isomorphic(pf6_graph.graph, pf6_graph2.graph, edge_match=em)


def test_molgraph_from_openff_mol_cco():
    from pymatgen.analysis.local_env import OpenBabelNN

    cco_openff = openff.toolkit.topology.Molecule.from_smiles("CCO")
    cco_openff.assign_partial_charges("mmff94")

    cco_molgraph_1 = molgraph_from_openff_mol(cco_openff)

    assert len(cco_molgraph_1.molecule) == 9
    assert cco_molgraph_1.molecule.charge == 0
    assert len(cco_molgraph_1.graph.edges) == 8

    cco_pmg = pymatgen.core.Molecule.from_file(CCO_xyz)
    cco_molgraph_2 = MoleculeGraph.with_local_env_strategy(cco_pmg, OpenBabelNN())

    em = iso.categorical_edge_match("weight", 1)

    assert nx.is_isomorphic(cco_molgraph_1.graph, cco_molgraph_2.graph, edge_match=em)


def test_openff_back_and_forth():

    cco_openff = openff.toolkit.topology.Molecule.from_smiles("CC(=O)O")
    cco_openff.assign_partial_charges("mmff94")

    cco_molgraph_1 = molgraph_from_openff_mol(cco_openff)

    assert len(cco_molgraph_1.molecule) == 8
    assert cco_molgraph_1.molecule.charge == 0
    assert len(cco_molgraph_1.graph.edges) == 7

    cco_openff_2 = molgraph_to_openff_mol(cco_molgraph_1)

    assert tk.Molecule.is_isomorphic_with(
        cco_openff, cco_openff_2, bond_order_matching=True
    )
    assert max(bond.bond_order for bond in cco_openff_2.bonds) == 2


def test_molgraph_to_openff_pf6():
    """transform a water MoleculeGraph to a OpenFF water molecule"""
    pf6_mol = pymatgen.core.Molecule.from_file(PF6_xyz)
    pf6_mol.set_charge_and_spin(charge=-1)
    pf6_molgraph = MoleculeGraph.with_edges(
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

    pf6_openff_1 = openff.toolkit.topology.Molecule.from_smiles("F[P-](F)(F)(F)(F)F")

    pf6_openff_2 = molgraph_to_openff_mol(pf6_molgraph)
    assert pf6_openff_1 == pf6_openff_2


def test_molgraph_to_openff_cco():
    from pymatgen.analysis.local_env import OpenBabelNN

    cco_pmg = pymatgen.core.Molecule.from_file(CCO_xyz)
    cco_molgraph = MoleculeGraph.with_local_env_strategy(cco_pmg, OpenBabelNN())

    cco_openff_1 = molgraph_to_openff_mol(cco_molgraph)

    cco_openff_2 = openff.toolkit.topology.Molecule.from_smiles("CCO")
    cco_openff_2.assign_partial_charges("mmff94")

    assert cco_openff_1 == cco_openff_2


def test_molgraph_from_openff_topology():
    O = tk.Molecule.from_smiles("O")
    CCO = tk.Molecule.from_smiles("CCO")

    topology = get_openff_topology({O: 200, CCO: 20})
    molgraph = molgraph_from_openff_topology(topology)

    subgraphs = molgraph.get_disconnected_fragments()

    # assert each molecule lengths in molgraph are the same as molecule lengths in topology
    mol_lengths = [len(mol.atoms) for mol in topology.molecules]
    molgraph_lengths = [len(subgraph.molecule) for subgraph in subgraphs]
    np.testing.assert_almost_equal(mol_lengths, molgraph_lengths)

    # test serialization
    json.dumps(molgraph, cls=MontyEncoder)
    # subgraphs = molgraph.get_disconnected_fragments()


def test_molgraph_to_openff_topology():
    O = tk.Molecule.from_smiles("O")
    CCO = tk.Molecule.from_smiles("CCO")

    topology = get_openff_topology({O: 2, CCO: 1})
    molgraph = molgraph_from_openff_topology(topology)

    topology_2, index_map = molgraph_to_openff_topology(molgraph, return_index_map=True)

    assert topology.n_atoms == topology_2.n_atoms
    assert topology.n_molecules == topology_2.n_molecules
    assert topology.n_bonds == topology_2.n_bonds

    molgraph.add_edge(0, 6)
    topology_3, _ = molgraph_to_openff_topology(molgraph, return_index_map=True)
    assert topology_3.n_molecules == 2
    assert topology_3.n_atoms == 15
    assert topology_3.n_bonds == 13
