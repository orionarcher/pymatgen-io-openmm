"""
Pytest fixtures for the testing pymatgen.io.openmm.
"""

import pytest

from pymatgen.io.openmm.alchemy_utils import AlchemicalReaction


@pytest.fixture
def acetic_rxn():
    """
    A fixture for an acetic acid and ethanol condensation Alchemical Reaction where
    the water is kept.
    """
    select_dict = {
        "acetic_acid_O": "smarts CC(=O)O and smarts C-O and element O",
        "acetic_acid_H": "smarts CC(=O)O[H] and smarts [OX2][H] and element H",
        "acetic_acid_C": "smarts CC(=O)O and smarts C=O and element C",
        "ethanol_O": "smarts [CH3][CH2]O and element O",
        "ethanol_H": "smarts [CH3][CH2]O[H] and smarts [OX2][H] and element H",
    }
    create_bonds = [("acetic_acid_C", "ethanol_O"), ("acetic_acid_O", "ethanol_H")]
    delete_bonds = [("ethanol_O", "ethanol_H"), ("acetic_acid_O", "acetic_acid_C")]
    return AlchemicalReaction(
        select_dict=select_dict,
        create_bonds=create_bonds,
        delete_bonds=delete_bonds,
    )


@pytest.fixture
def acetic_rxn_rm_water():
    """
    A fixture for an acetic acid and ethanol condensation Alchemical Reaction where
    the water is deleted.
    """
    select_dict = {
        "acetic_acid_O": "smarts CC(=O)O and smarts C-O and element O",
        "acetic_acid_H": "smarts CC(=O)O[H] and smarts [OX2][H] and element H",
        "acetic_acid_C": "smarts CC(=O)O and smarts C=O and element C",
        "ethanol_O": "smarts [CH3][CH2]O and element O",
        "ethanol_H": "smarts [CH3][CH2]O[H] and smarts [OX2][H] and element H",
    }
    create_bonds = [("acetic_acid_C", "ethanol_O")]
    delete_atoms = ["acetic_acid_O", "acetic_acid_H", "ethanol_H"]
    return AlchemicalReaction(
        select_dict=select_dict,
        create_bonds=create_bonds,
        delete_atoms=delete_atoms,
    )


@pytest.fixture
def propanedioic_rxn_rm_water():
    """
    A fixture for an acetic acid and ethanol condensation Alchemical Reaction where
    the water is deleted.
    """
    select_dict = {
        "propanedioic_acid_O": "smarts OC(=O)CC(O)=O and smarts C-O and element O",
        "propanedioic_acid_H": "(byres smarts OC(=O)CC(O)=O) and smarts [OX2][H] and element H",
        "propanedioic_acid_C": "smarts OC(=O)CC(O)=O and smarts C=O and element C",
        "ethanol_O": "smarts [CH3][CH2]O and element O",
        "ethanol_H": "smarts [CH3][CH2]O[H] and smarts [OX2][H] and element H",
    }
    create_bonds = [("propanedioic_acid_C", "ethanol_O")]
    delete_atoms = ["propanedioic_acid_O", "propanedioic_acid_H", "ethanol_H"]
    return AlchemicalReaction(
        select_dict=select_dict,
        create_bonds=create_bonds,
        delete_atoms=delete_atoms,
    )
