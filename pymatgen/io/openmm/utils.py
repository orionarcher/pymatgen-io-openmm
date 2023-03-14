"""
Utility functions for OpenMM simulation setup.
"""
from typing import Dict, Iterable, List, Tuple, Union
import pathlib
from pathlib import Path
import tempfile

import numpy as np
import rdkit
import openff
import openff.toolkit as tk
from openmm.unit import elementary_charge, angstrom
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Element, Molecule

import pymatgen
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.xyz import XYZ
from pymatgen.io.packmol import PackmolBoxGen
from openmm.app import Topology as OpenMMTopology

from pint import Quantity


def smiles_to_atom_type_array(openff_counts: Dict[tk.Molecule, int]) -> np.ndarray:
    """
    Convert a SMILE to an array of atom types.

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
        print("in tempfile")
        pw = PackmolBoxGen(seed=random_seed).get_input_set(
            molecules: list(Molecule)=packmol_molecules, box=box
        )

        print("here")
        pw.write_input(scratch_dir)
        pw.run(scratch_dir, timeout=packmol_timeout)
        coordinates = XYZ.from_file(
            pathlib.Path(scratch_dir, "packmol_out.xyz")
        ).as_dataframe()
    raw_coordinates = coordinates.loc[:, "x":"z"].values  # type: ignore
    return raw_coordinates

def molgraph_from_atoms_bonds(
    atoms: Iterable[openff.toolkit.topology.Atom],
    bonds: Iterable[openff.toolkit.topology.Bond],
    coords: Union[None, np.ndarray] = None,
    name: str = "none",
):
    """
    This is designed to closely mirror the graph structure generated by tk.Molecule.to_networkx
    """
    molgraph = MoleculeGraph.with_empty_graph(
        Molecule([], []),
        name=name,
    )
    p_table = {el.Z: str(el) for el in Element}
    formal_charge = 0
    for i, atom in enumerate(atoms):
        molgraph.insert_node(
            i,
            p_table[atom.atomic_number],
            coords[i, :] if coords is not None else np.array([0, 0, 0]),
        )
        molgraph.graph.nodes[i]["atomic_number"] = atom.atomic_number
        molgraph.graph.nodes[i]["is_aromatic"] = atom.is_aromatic
        molgraph.graph.nodes[i]["stereochemistry"] = atom.stereochemistry
        molgraph.graph.nodes[i]["formal_charge"] = atom.formal_charge
        molgraph.graph.nodes[i]["partial_charge"] = atom.partial_charge
        formal_charge += atom.formal_charge.magnitude

    molgraph.molecule.set_charge_and_spin(charge=formal_charge)

    for bond in bonds:
        molgraph.graph.add_edge(
            bond.atom1_index,
            bond.atom2_index,
            bond_order=bond.bond_order,
            is_aromatic=bond.is_aromatic,
            stereochemistry=bond.stereochemistry,
        )

    return molgraph


def openmm_topology_from_molgraph(molgraph):
    return


# def openff_topology_from_molgraph(molgraph, unique_molecules=None):
#     """
#     Construct an OpenFF Topology object from an OpenMM Topology object.
#
#     Parameters
#     ----------
#     openmm_topology : simtk.openmm.app.Topology
#         An OpenMM Topology object
#     unique_molecules : iterable of objects that can be used to construct unique Molecule objects
#         All unique molecules must be provided, in any order, though multiple copies of each molecule are allowed.
#         The atomic elements and bond connectivity will be used to match the reference molecules
#         to molecule graphs appearing in the OpenMM ``Topology``. If bond orders are present in the
#         OpenMM ``Topology``, these will be used in matching as well.
#
#     Returns
#     -------
#     topology : openff.toolkit.topology.Topology
#         An OpenFF Topology object
#     """
#     # credit to source code from openff.toolkit.topology.topology
#     import networkx as nx
#
#     from openff.toolkit.topology.molecule import Molecule
#
#     # Check to see if the openMM system has defined bond orders, by looping over all Bonds in the Topology.
#     omm_has_bond_orders = True
#     for omm_bond in openmm_topology.bonds():
#         if omm_bond.order is None:
#             omm_has_bond_orders = False
#
#     if unique_molecules is None:
#         raise MissingUniqueMoleculesError(
#             "Topology.from_openmm requires a list of Molecule objects "
#             "passed as unique_molecules, but None was passed."
#         )
#
#     # Convert all unique mols to graphs
#     topology = cls()
#     graph_to_unq_mol = {}
#     for unq_mol in unique_molecules:
#         unq_mol_graph = unq_mol.to_networkx()
#         for existing_graph in graph_to_unq_mol.keys():
#             if Molecule.are_isomorphic(
#                     existing_graph,
#                     unq_mol_graph,
#                     return_atom_map=False,
#                     aromatic_matching=False,
#                     formal_charge_matching=False,
#                     bond_order_matching=omm_has_bond_orders,
#                     atom_stereochemistry_matching=False,
#                     bond_stereochemistry_matching=False,
#             )[0]:
#                 msg = (
#                     "Error: Two unique molecules have indistinguishable "
#                     "graphs: {} and {}".format(
#                         unq_mol, graph_to_unq_mol[existing_graph]
#                     )
#                 )
#                 raise DuplicateUniqueMoleculeError(msg)
#         graph_to_unq_mol[unq_mol_graph] = unq_mol
#
#     # Convert all openMM mols to graphs
#     omm_topology_G = nx.Graph()
#     for atom in openmm_topology.atoms():
#         omm_topology_G.add_node(atom.index, atomic_number=atom.element.atomic_number)
#     for bond in openmm_topology.bonds():
#         omm_topology_G.add_edge(
#             bond.atom1.index, bond.atom2.index, bond_order=bond.order
#         )
#
#     # For each connected subgraph (molecule) in the topology, find its match in unique_molecules
#     topology_molecules_to_add = list()
#     for omm_mol_G in (
#             omm_topology_G.subgraph(c).copy()
#             for c in nx.connected_components(omm_topology_G)
#     ):
#         match_found = False
#         for unq_mol_G in graph_to_unq_mol.keys():
#             isomorphic, mapping = Molecule.are_isomorphic(
#                 omm_mol_G,
#                 unq_mol_G,
#                 return_atom_map=True,
#                 aromatic_matching=False,
#                 formal_charge_matching=False,
#                 bond_order_matching=omm_has_bond_orders,
#                 atom_stereochemistry_matching=False,
#                 bond_stereochemistry_matching=False,
#             )
#             if isomorphic:
#                 # Take the first valid atom indexing map
#                 first_topology_atom_index = min(mapping.keys())
#                 topology_molecules_to_add.append(
#                     (first_topology_atom_index, unq_mol_G, mapping.items())
#                 )
#                 match_found = True
#                 break
#         if match_found is False:
#             hill_formula = Molecule.to_hill_formula(omm_mol_G)
#             msg = f"No match found for molecule {hill_formula}. "
#             probably_missing_conect = [
#                 "C",
#                 "H",
#                 "O",
#                 "N",
#                 "P",
#                 "S",
#                 "F",
#                 "Cl",
#                 "Br",
#             ]
#             if hill_formula in probably_missing_conect:
#                 msg += (
#                     "This would be a very unusual molecule to try and parameterize, "
#                     "and it is likely that the data source it was read from does not "
#                     "contain connectivity information. If this molecule is coming from "
#                     "PDB, please ensure that the file contains CONECT records. The PDB "
#                     "format documentation (https://www.wwpdb.org/documentation/"
#                     'file-format-content/format33/sect10.html) states "CONECT records '
#                     "are mandatory for HET groups (excluding water) and for other bonds "
#                     'not specified in the standard residue connectivity table."'
#                 )
#             raise ValueError(msg)
#
#     # The connected_component_subgraph function above may have scrambled the molecule order, so sort molecules
#     # by their first atom's topology index
#     topology_molecules_to_add.sort(key=lambda x: x[0])
#     for first_index, unq_mol_G, top_to_ref_index in topology_molecules_to_add:
#         local_top_to_ref_index = {
#             top_index - first_index: ref_index
#             for top_index, ref_index in top_to_ref_index
#         }
#         topology.add_molecule(
#             graph_to_unq_mol[unq_mol_G],
#             local_topology_to_reference_index=local_top_to_ref_index,
#         )
#
#     topology.box_vectors = openmm_topology.getPeriodicBoxVectors()
#     # TODO: How can we preserve metadata from the openMM topology when creating the OFF topology?
#     return topology


def molgraph_to_openff_mol(molgraph: MoleculeGraph) -> tk.Molecule:
    """
    Convert a Pymatgen MoleculeGraph to an OpenFF Molecule.

    If partial charges, formal charges, and aromaticity are present in site properties
    they will be mapped onto atoms.
    If bond order and bond aromaticity are present in edge weights and edge properties
    they will be mapped onto bonds.

    Args:
        openff_mol: OpenFF Molecule

    Returns:
        MoleculeGraph
    """
    # create empty openff_mol and prepare a periodic table
    p_table = {str(el): el.Z for el in Element}
    openff_mol = openff.toolkit.topology.Molecule()

    # set atom properties
    partial_charges = []
    for i_node in molgraph.graph.nodes:
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
    # set up coords and species
    if openff_mol.n_conformers > 0:
        coords = openff_mol.conformers[0] / angstrom
    else:
        coords = np.zeros((openff_mol.n_atoms, 3))

    molgraph = molgraph_from_atoms_bonds(
        openff_mol.atoms, openff_mol.bonds, coords=coords, name=openff_mol.name
    )
    return molgraph


def molgraph_from_openff_topology(topology: tk.Topology):
    molgraph = molgraph_from_atoms_bonds(topology.atoms, topology.bonds)
    return molgraph


def get_openff_topology(openff_counts: Dict[tk.Molecule, int]) -> tk.Topology:
    """
    Returns an openff topology with the given SMILEs at the given counts.

    Parameters:
        smiles: keys are smiles and values are number of that molecule to pack

    Returns:
        an openmm.app.Topology
    """
    mols = []
    for mol, count in openff_counts.items():
        mols += [mol] * count
    return tk.topology.Topology.from_molecules(mols)


def infer_openff_mol(
    mol_geometry: pymatgen.core.Molecule,
) -> tk.Molecule:
    """
    Infer an OpenFF molecule from xyz coordinates.
    """
    # TODO: we can just have Molecule Graph be the only internal representation
    with tempfile.NamedTemporaryFile() as f:
        # TODO: allow for Molecule graphs
        # TODO: build a MoleculeGraph -> OpenFF mol direct converter
        # these next 4 lines are cursed
        pybel_mol = BabelMolAdaptor(mol_geometry).pybel_mol  # pymatgen Molecule
        pybel_mol.write("mol2", filename=f.name, overwrite=True)  # pybel Molecule
        rdmol = rdkit.Chem.MolFromMol2File(f.name, removeHs=False)  # rdkit Molecule
    inferred_mol = openff.toolkit.topology.Molecule.from_rdkit(
        rdmol, hydrogens_are_explicit=True
    )  # OpenFF Molecule
    return inferred_mol
