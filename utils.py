"""
Utility functions for OpenMM simulation setup.
"""
from typing import Dict, List
import tempfile

import numpy as np
from openbabel import pybel
import parmed

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
    given density. The corner of the cube is at the origin.

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


def n_mols_from_mass_ratio(n_mol: int, smiles: List[str], mass_ratio: List[float]) -> np.ndarray[int]:
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
) -> np.ndarray[int]:
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
