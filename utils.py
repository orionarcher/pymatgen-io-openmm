"""
Utility functions for OpenMM simulation setup.
"""
from typing import Dict, List, Union, Tuple
from pathlib import Path
import tempfile

import numpy as np
from openbabel import pybel
import parmed
import rdkit
import openff

import pymatgen
from pymatgen.io.babel import BabelMolAdaptor


def smile_to_parmed_structure(smile: str) -> parmed.Structure:
    """
    Convert a SMILE to a Parmed.Structure.
    """
    mol = pybel.readstring("smi", smile)
    mol.addh()
    mol.make3D()
    with tempfile.NamedTemporaryFile() as f:
        mol.write(format="mol", filename=f.name, overwrite=True)
        structure = parmed.load_file(f.name)[0]  # load_file is returning a list for some reason
    return structure


def smile_to_molecule(smile: str) -> pymatgen.core.Molecule:
    """
    Convert a SMILE to a Pymatgen.Molecule.
    """
    mol = pybel.readstring("smi", smile)
    mol.addh()
    mol.make3D()
    adaptor = BabelMolAdaptor(mol.OBMol)
    return adaptor.pymatgen_mol


def get_box(smiles: Dict[str, int], density: float) -> List[float]:
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
    mols = [smile_to_molecule(smile) for smile in smiles.keys()]
    mol_mw = np.array([mol.composition.weight for mol in mols])
    counts = np.array(list(smiles.values()))
    total_weight = sum(mol_mw * counts)
    box_volume = total_weight * cm3_to_A3 / (NA * density)
    side_length = round(box_volume ** (1 / 3), 2)
    return [0, 0, 0, side_length, side_length, side_length]


def n_mols_from_mass_ratio(n_mol: int, smiles: List[str], mass_ratio: List[float]) -> np.ndarray:
    """
    Calculates the number of mols needed to yield a given mass ratio.

    Args:
        n_mol: total number of mols. returned array will sum to n_mol. e.g. sum(n_mols) = n_mol.
        smiles: a list of smiles. e.g. ["O", "CCO"]
        mass_ratio: mass ratio of smiles. e.g. [9, 1]

    Returns:
        n_mols: number of each smile needed for mass ratio.
    """
    mols = [smile_to_molecule(smile) for smile in smiles]
    mws = np.array([mol.composition.weight for mol in mols])
    mol_ratio = np.array(mass_ratio) / mws
    mol_ratio /= sum(mol_ratio)
    return np.round(mol_ratio * n_mol)


def n_mols_from_volume_ratio(
    n_mol: int, smiles: List[str], volume_ratio: List[float], densities: List[float]
) -> np.ndarray:
    """
    Calculates the number of mols needed to yield a given volume ratio.

    Args:
        n_mol: total number of mols. returned array will sum to n_mol. e.g. sum(n_mols) = n_mol.
        smiles: a list of smiles. e.g. ["O", "CCO"]
        volume_ratio: volume ratio of smiles. e.g. [9, 1]
        densities: density of each smile. e.g. [1, 0.79]

    Returns:
        n_mols: number of each smile needed for volume ratio.

    """
    mass_ratio = np.array(volume_ratio) * np.array(densities)
    return n_mols_from_mass_ratio(n_mol, smiles, mass_ratio)


def n_solute_from_molarity(molarity: float, volume: float) -> int:
    """
    Calculates the number of solute molecules needed for a given molarity.

    Args:
        molarity: molarity of solute desired.
        volume: volume of box in liters.

    Returns:
        n_solute: number of solute molecules

    """
    NA = 6.02214e23
    n_solute = volume * NA * molarity
    return round(n_solute)


def calculate_molarity(volume, n_solute):
    """
    Calculate the molarity of a number of solutes in a volume.
    """
    NA = 6.02214e23
    molarity = n_solute / (volume * NA)
    return molarity


def get_atom_map(inferred_mol, openff_mol) -> Tuple[bool, Dict[int, int]]:
    """
    Get a mapping between two openff Molecules.
    """
    # do not apply formal charge restrictions
    kwargs = dict(
        return_atom_map=True,
        formal_charge_matching=False,
    )
    isomorphic, atom_map = openff.toolkit.topology.Molecule.are_isomorphic(openff_mol, inferred_mol, **kwargs)
    if isomorphic:
        return True, atom_map
    # relax stereochemistry restrictions
    kwargs["atom_stereochemistry_matching"] = False
    kwargs["bond_stereochemistry_matching"] = False
    isomorphic, atom_map = openff.toolkit.topology.Molecule.are_isomorphic(openff_mol, inferred_mol, **kwargs)
    if isomorphic:
        print(f"stereochemistry ignored when matching inferred" f"mol: {openff_mol} to {inferred_mol}")
        return True, atom_map
    # relax bond order restrictions
    kwargs["bond_order_matching"] = False
    isomorphic, atom_map = openff.toolkit.topology.Molecule.are_isomorphic(openff_mol, inferred_mol, **kwargs)
    if isomorphic:
        print(f"stereochemistry ignored when matching inferred" f"mol: {openff_mol} to {inferred_mol}")
        print(f"bond_order restrictions ignored when matching inferred" f"mol: {openff_mol} to {inferred_mol}")
        return True, atom_map
    return False, {}


def infer_openff_mol(mol_geometry: Union[pymatgen.core.Molecule, str, Path]) -> openff.toolkit.topology.Molecule:
    """
    Infer an OpenFF molecule from xyz coordinates.
    """
    if isinstance(mol_geometry, (str, Path)):
        mol_geometry = pymatgen.core.Molecule.from_file(str(mol_geometry))
    with tempfile.NamedTemporaryFile() as f:
        # these next 4 lines are cursed
        pybel_mol = BabelMolAdaptor(mol_geometry).pybel_mol  # pymatgen Molecule
        pybel_mol.write("mol2", filename=f.name, overwrite=True)  # pybel Molecule
        rdmol = rdkit.Chem.MolFromMol2File(f.name, removeHs=False)  # rdkit Molecule
    inferred_mol = openff.toolkit.topology.Molecule.from_rdkit(rdmol, hydrogens_are_explicit=True)  # OpenFF Molecule
    return inferred_mol


def order_molecule_like_smile(smile: str, geometry: Union[pymatgen.core.Molecule, str, Path]):
    """
    Order sites in a pymatgen Molecule to match the canonical ordering generated by rdkit.
    """
    if isinstance(geometry, (str, Path)):
        geometry = pymatgen.core.Molecule.from_file(str(geometry))
    inferred_mol = infer_openff_mol(geometry)
    openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
    is_isomorphic, atom_map = get_atom_map(inferred_mol, openff_mol)
    new_molecule = pymatgen.core.Molecule.from_sites([geometry.sites[i] for i in atom_map.values()])
    return new_molecule
