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
    Calculates the side_length of a cube necessary to contain the given molecules with
    given density.

    Args:
        smiles: keys are smiles and values are number of that molecule to pack
        density: guessed density of the solution, larger densities will lead to smaller cubes.

    Returns:
        side_length: side length of the returned cube
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
