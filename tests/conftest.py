"""
Pytest fixtures for the testing pymatgen.io.openmm.
"""

import pytest

from pymatgen.io.openmm.alchemy_utils import AlchemicalReaction


@pytest.fixture
def select_dict():
    """
    A fixture for a select dict for an acetic acid and ethanol condensation Alchemical Reaction.
    """
    select_dict = {
        "acetic_acid_O": "smarts CC(=O)O and smarts C-O and element O",
        "acetic_acid_H": "smarts CC(=O)O[H] and smarts [OX2][H] and element H",
        "acetic_acid_C": "smarts CC(=O)O and smarts C=O and element C",
        "ethanol_O": "smarts [CH3][CH2]O and element O",
        "ethanol_H": "smarts [CH3][CH2]O[H] and smarts [OX2][H] and element H",
    }
    return select_dict


@pytest.fixture
def acetic_ethanol_condensation(select_dict):
    """
    A fixture for an acetic acid and ethanol condensation Alchemical Reaction where
    the water is kept.
    """
    create_bonds = [("acetic_acid_C", "ethanol_O"), ("acetic_acid_O", "ethanol_H")]
    delete_bonds = [("ethanol_O", "ethanol_H"), ("acetic_acid_O", "acetic_acid_C")]
    rxn = AlchemicalReaction(
        select_dict=select_dict,
        create_bonds=create_bonds,
        delete_bonds=delete_bonds,
    )
    return rxn


@pytest.fixture
def acetic_ethanol_condensation_del_water(select_dict):
    """
    A fixture for an acetic acid and ethanol condensation Alchemical Reaction where
    the water is deleted.
    """
    create_bonds = [("acetic_acid_C", "ethanol_O")]
    delete_atoms = ["acetic_acid_O", "acetic_acid_H", "ethanol_H"]
    rxn = AlchemicalReaction(
        select_dict=select_dict,
        create_bonds=create_bonds,
        delete_atoms=delete_atoms,
    )
    return rxn
