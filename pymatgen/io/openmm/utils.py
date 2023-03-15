"""
Utility functions for OpenMM simulation setup.
"""
from typing import Dict, Iterable, List, Tuple, Union
import pathlib
from pathlib import Path
import tempfile

import numpy as np
import openff
import openff.toolkit as tk
from openmm.unit import elementary_charge, angstrom
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Element, Molecule
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender

import pymatgen
from pymatgen.io.xyz import XYZ
from pymatgen.io.packmol import PackmolBoxGen

from pint import Quantity


def smiles_to_atom_type_array(openff_counts: Dict[tk.Molecule, int]) -> np.ndarray:
    """
    Cal to an array of atom types.

    Args:
        smiles:

    Returns:

    """
    offset = 0
    all_types_list = []
    for mol, count in openff_counts.items():
        types = np.arange(offset, offset + mol.n_atoms)
        types_array = np.hstack([types for _ in range(count)])
        all_types_list.append(types_array)
        offset += mol.n_atoms
    return np.concatenate(all_types_list)


def smiles_to_resname_array(
    mol_specs: List[Dict[str, Union[str, int, tk.Molecule]]]
) -> np.ndarray:
    """
    Convert a list of SMILEs to an array of residue names.

    Args:
        smiles:
        names: dictionary of residue names for each smile, where keys are smiles
        and values are residue names. If not provided, residue names will be set
        to the smile.

    Returns:
        resname_array: array of residue names.
    """
    resnames = []
    for mol_spec in mol_specs:
        name = mol_spec["name"]
        count = mol_spec["count"]
        smile_size = mol_spec["openff_mol"].n_atoms  # type: ignore
        resnames.extend([name] * count * smile_size)  # type: ignore
    return np.array(resnames)


def xyz_to_molecule(
    mol_geometry: Union[pymatgen.core.Molecule, str, Path]
) -> pymatgen.core.Molecule:
    """
    Convert a XYZ file to a Pymatgen.Molecule.

    Accepts a str or pathlib.Path file that can be parsed for xyz coordinates from OpenBabel and
    returns a Pymatgen.Molecule. If a Pymatgen.Molecule is passed in, it is returned unchanged.

    Args:
        mol_geometry:

    Returns:

    """
    if isinstance(mol_geometry, (str, Path)):
        mol_geometry = pymatgen.core.Molecule.from_file(str(mol_geometry))
    return mol_geometry


def get_box(openff_counts: Dict[tk.Molecule, int], density: float) -> List[float]:
    """
    Calculates the dimensions of a cube necessary to contain the given molecules with
    given density. The corner of the cube is at the origin. Units are angstrom.

    Args:
        smiles: keys are smiles and values are number of that molecule to pack
        density: guessed density of the solution, larger densities will lead to smaller cubes.

    Returns:
        dimensions: array of [0, 0, 0, side_length, side_length, side_length]
    """
    cm3_to_A3 = 1e24
    NA = 6.02214e23

    def get_mass(mol):
        return sum(atom.mass.magnitude for atom in mol.atoms)

    mol_mw = np.array([get_mass(mol) for mol in openff_counts.keys()])
    counts = np.array(list(openff_counts.values()))
    total_weight = sum(mol_mw * counts)
    box_volume = total_weight * cm3_to_A3 / (NA * density)
    side_length = round(box_volume ** (1 / 3), 2)
    return [0, 0, 0, side_length, side_length, side_length]


def get_atom_map(inferred_mol, openff_mol) -> Tuple[bool, Dict[int, int]]:
    """
    Get a mapping between two openff Molecules.
    """
    # do not apply formal charge restrictions
    kwargs = dict(
        return_atom_map=True,
        formal_charge_matching=False,
    )
    isomorphic, atom_map = openff.toolkit.topology.Molecule.are_isomorphic(
        openff_mol, inferred_mol, **kwargs
    )
    if isomorphic:
        return True, atom_map
    # relax stereochemistry restrictions
    kwargs["atom_stereochemistry_matching"] = False
    kwargs["bond_stereochemistry_matching"] = False
    isomorphic, atom_map = openff.toolkit.topology.Molecule.are_isomorphic(
        openff_mol, inferred_mol, **kwargs
    )
    if isomorphic:
        print(
            f"stereochemistry ignored when matching inferred"
            f"mol: {openff_mol} to {inferred_mol}"
        )
        return True, atom_map
    # relax bond order restrictions
    kwargs["bond_order_matching"] = False
    isomorphic, atom_map = openff.toolkit.topology.Molecule.are_isomorphic(
        openff_mol, inferred_mol, **kwargs
    )
    if isomorphic:
        print(
            f"stereochemistry ignored when matching inferred"
            f"mol: {openff_mol} to {inferred_mol}"
        )
        print(
            f"bond_order restrictions ignored when matching inferred"
            f"mol: {openff_mol} to {inferred_mol}"
        )
        return True, atom_map
    return False, {}


def get_coordinates(
    openff_counts: Dict[tk.Molecule, int],
    box: List[float],
    random_seed: int = -1,
    packmol_timeout: int = 30,
) -> np.ndarray:
    """
    Pack the box with the molecules specified by smiles.

    Args:
        openff_counts: keys are openff molecules and values are number of that molecule to pack.
            The molecules must already have conformations.
        box: list of [xlo, ylo, zlo, xhi, yhi, zhi]
        random_seed: the random seed used by packmol
        smile_geometries: a dictionary of smiles and their respective geometries. The
            geometries must be pymatgen Molecules.
        packmol_timeout: the number of seconds to wait for packmol to finish before
            raising an Error.

    Returns:
        array of coordinates for each atom in the box.
    """
    from pymatgen.core import Site

    packmol_molecules = []
    for i, mol_count in enumerate(openff_counts.items()):
        mol, count = mol_count
        n_conformers = len(mol.conformers)
        atomic_numbers = [atom.atomic_number for atom in mol.atoms]
        for j, conformer in enumerate(mol.conformers):
            sites = [
                Site(atomic_numbers[j], conformer.magnitude[j, :])
                for j in range(mol.n_atoms)
            ]
            coords = Molecule.from_sites(sites)
            # TODO: test the units here
            packmol_molecules.append(
                {
                    "name": f"mol_{i}_conformer_{j}",
                    "number": count // n_conformers + int(count % n_conformers > j),
                    "coords": coords,
                }
            )
    with tempfile.TemporaryDirectory() as scratch_dir:
        pw = PackmolBoxGen(seed=random_seed).get_input_set(
            molecules=packmol_molecules, box=box
        )
        pw.write_input(scratch_dir)
        pw.run(scratch_dir, timeout=packmol_timeout)
        coordinates = XYZ.from_file(
            pathlib.Path(scratch_dir, "packmol_out.xyz")
        ).as_dataframe()
    raw_coordinates = coordinates.loc[:, "x":"z"].values  # type: ignore
    return raw_coordinates


def parameterize_w_interchange(openff_topology, mol_specs, box, force_field="sage"):
    assert force_field == "sage", "currently only the sage force field is supported"

    from openff.interchange import Interchange
    from openff.toolkit import ForceField
    import numpy as np

    box_arr = np.array(box)
    box_matrix = np.diag(box_arr[3:6] - box_arr[0:3]) * angstrom
    sage = ForceField("openff_unconstrained-2.0.0.offxml")
    interchange = Interchange.from_smirnoff(
        force_field=sage,
        topology=openff_topology,
        charge_from_molecules=[spec["openff_mol"] for spec in mol_specs],
        box=box_matrix,
        allow_nonintegral_charges=True,
    )
    return interchange.to_openmm()


def molgraph_from_molecules(molecules: Iterable[tk.Molecule]):
    """
    This is designed to closely mirror the graph structure generated by tk.Molecule.to_networkx
    """
    molgraph = MoleculeGraph.with_empty_graph(
        Molecule([], []),
        name="none",
    )
    p_table = {el.Z: str(el) for el in Element}
    formal_charge = 0
    cum_atoms = 0
    for molecule in molecules:
        if molecule.conformers is not None:
            coords = molecule.conformers[0].magnitude
        else:
            coords = np.zeros((molecule.n_atoms, 3))
        for j, atom in enumerate(molecule.atoms):
            molgraph.insert_node(
                cum_atoms + j,
                p_table[atom.atomic_number],
                coords[j, :],
            )
            molgraph.graph.nodes[cum_atoms + j]["atomic_number"] = atom.atomic_number
            molgraph.graph.nodes[cum_atoms + j]["is_aromatic"] = atom.is_aromatic
            molgraph.graph.nodes[cum_atoms + j][
                "stereochemistry"
            ] = atom.stereochemistry
            molgraph.graph.nodes[cum_atoms + j]["formal_charge"] = atom.formal_charge
            molgraph.graph.nodes[cum_atoms + j]["partial_charge"] = atom.partial_charge
            formal_charge += atom.formal_charge.magnitude
        for bond in molecule.bonds:
            molgraph.graph.add_edge(
                cum_atoms + bond.atom1_index,
                cum_atoms + bond.atom2_index,
                bond_order=bond.bond_order,
                is_aromatic=bond.is_aromatic,
                stereochemistry=bond.stereochemistry,
            )
        # formal_charge += molecule.total_charge
        cum_atoms += molecule.n_atoms
    molgraph.molecule.set_charge_and_spin(charge=formal_charge)
    return molgraph


def get_unique_subgraphs(molgraph_list: List[MoleculeGraph]) -> List[MoleculeGraph]:
    """
    This function takes a list of molecule graphs and returns a list of unique subgraphs.
    It uses the weisfeiler_lehman_graph_hash neworkx.algorithms.graph_hashing to create a
    dictionary of unique graphs.
    """
    import networkx as nx
    from copy import deepcopy

    # create a dictionary of unique graphs
    unique_graphs = {}
    for molgraph in molgraph_list:
        # TODO: currently does not use bond order to hash
        graph_hash = nx.weisfeiler_lehman_graph_hash(
            molgraph.graph,
            # edge_attr='bond_order',
            node_attr="atomic_number",
        )
        unique_graphs[graph_hash] = deepcopy(molgraph)
    return list(unique_graphs.values())


def molgraph_to_openff_topology(molgraph, return_index_map=False):
    """
    Convert a Pymatgen MoleculeGraph to an OpenFF Topology.

    The atom ordering in the OpenFF topology may differ from the atom ordering
    in the Pymatgen MoleculeGraph. The new_to_old_index dictionary maps the
    new atom ordering to the old atom ordering.

    There is no guarantee that atom positions will be preserved.

    Args:
        molgraph: A Pymatgen MoleculeGraph
        return_index_map:
            If True, return a dictionary mapping the new atom ordering to the old atom ordering.

    Returns:
        OpenFF Topology, new_to_old_index


    """
    # dependent on pmg addition
    subgraphs, new_to_old_index = molgraph.get_disconnected_fragments(
        return_index_map=True
    )
    molecules = [molgraph_to_openff_mol(subgraph) for subgraph in subgraphs]
    openff_topology = tk.Topology.from_molecules(molecules)
    if return_index_map:
        return openff_topology, new_to_old_index
    return openff_topology


def molgraph_to_openff_mol(molgraph: MoleculeGraph) -> tk.Molecule:
    """
    Convert a Pymatgen MoleculeGraph to an OpenFF Molecule.

    If partial charges, formal charges, and aromaticity are present in site properties
    they will be mapped onto atoms. If bond order and bond aromaticity are present in
    edge weights and edge properties they will be mapped onto bonds.

    Args:
        molgraph: PyMatGen MoleculeGraph

    Returns:
        openff_mol: OpenFF Molecule
    """
    # create empty openff_mol and prepare a periodic table
    p_table = {str(el): el.Z for el in Element}
    openff_mol = openff.toolkit.topology.Molecule()

    # set atom properties
    partial_charges = []
    # TODO: should assert that there is only one molecule
    for i_node in range(len(molgraph.graph.nodes)):
        node = molgraph.graph.nodes[i_node]
        atomic_number = (
            node.get("atomic_number")
            or p_table[molgraph.molecule[i_node].species_string]
        )

        # put formal charge on first atom if there is none present
        formal_charge = node.get("formal_charge")
        if formal_charge is None:
            formal_charge = (i_node == 0) * molgraph.molecule.charge * elementary_charge

        # assume not aromatic if no info present
        is_aromatic = node.get("is_aromatic") or False

        openff_mol.add_atom(atomic_number, formal_charge, is_aromatic=is_aromatic)

        # add to partial charge array
        partial_charge = node.get("partial_charge")
        if isinstance(partial_charge, Quantity):
            partial_charge = partial_charge.magnitude
        partial_charges.append(partial_charge)

    charge_array = np.array(partial_charges)
    if np.not_equal(charge_array, None).all():
        openff_mol.partial_charges = charge_array * elementary_charge

    # set edge properties, default to single bond and assume not aromatic
    for i_node, j, bond_data in molgraph.graph.edges(data=True):
        bond_order = bond_data.get("weight", 1) or 1
        is_aromatic = bond_data.get("is_aromatic") or False
        openff_mol.add_bond(i_node, j, bond_order, is_aromatic=is_aromatic)

    openff_mol.add_conformer(molgraph.molecule.cart_coords * angstrom)
    return openff_mol


def molgraph_from_openff_mol(openff_mol: tk.Molecule) -> MoleculeGraph:
    """
    Convert an OpenFF Molecule to a Pymatgen MoleculeGraph.

    Preserves partial charges, formal charges, and aromaticity in site properties.
    Preserves bond order in edge weights and bond aromaticity in edge properties.

    Args:
        openff_mol: OpenFF Molecule

    Returns:
        MoleculeGraph
    """
    return molgraph_from_molecules([openff_mol])


def molgraph_from_openff_topology(topology: tk.Topology):
    return molgraph_from_molecules(topology.molecules)


def get_openff_topology(openff_counts: Dict[tk.Molecule, int]) -> tk.Topology:
    """
    Returns an openff topology with the given openff molecules at the given counts.

    Parameters:
        smiles: a dictionary of openff molecules and their counts

    Returns:
        an openff Topology
    """
    mols = []
    for mol, count in openff_counts.items():
        mols += [mol] * count
    return tk.topology.Topology.from_molecules(mols)


def infer_openff_mol(
    mol_geometry: pymatgen.core.Molecule,
) -> tk.Molecule:
    """
    Infer an OpenFF molecule from a pymatgen Molecule.

    Args:
        mol_geometry: A pymatgen Molecule

    Returns:
        an OpenFF Molecule

    """
    molgraph = MoleculeGraph.with_local_env_strategy(mol_geometry, OpenBabelNN())
    molgraph = metal_edge_extender(molgraph)
    inferred_mol = molgraph_to_openff_mol(molgraph)
    return inferred_mol


"""
Plan:
1. pack mol with geometries generated from openbabel
2. create an appropriate topology file
3. infer bonding with OpenBabel, THIS SHOULD WORK
4. test that it works with an array of smiles
5. test adding bonds and removing bonds
6. test regenerating PDB
7. test regenerating unique mols
8. implement and loop!
"""


def topology_to_universe(topology: tk.Topology):
    """
    probably no longer needed.
    """
    return


def get_atom_ix_in_universe(universe, select_strings):
    """
    given a universe and select strings, this will return the indices of the selected atoms.
    """
    return
