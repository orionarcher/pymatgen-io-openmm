import pytest

from pymatgen.io.openmm.alchemy_utils import AlchemicalReaction, smiles_to_universe


@pytest.fixture
def select_dict():
    select_dict = {
        "acetic_acid_O": "smarts CC(=O)O and smarts C-O and element O",
        "acetic_acid_H": "smarts CC(=O)O[H] and smarts [OX2][H] and element H",
        "acetic_acid_C": "smarts CC(=O)O and smarts C=O and element C",
        "ethanol_O": "smarts [CH3][CH2]O and element O",
        "ethanol_H": "smarts [CH3][CH2]O[H] and smarts [OX2][H] and element H",
    }
    return select_dict


@pytest.fixture
def acetic_ethanol_hydrolysis(select_dict):
    create_bonds = [("acetic_acid_C", "ethanol_O"), ("acetic_acid_O", "ethanol_H")]
    delete_bonds = [("ethanol_O", "ethanol_H"), ("acetic_acid_O", "acetic_acid_C")]
    rxn = AlchemicalReaction(
        select_dict=select_dict,
        create_bonds=create_bonds,
        delete_bonds=delete_bonds,
    )
    return rxn


@pytest.fixture
def acetic_ethanol_hydrolysis_del_water(select_dict):
    create_bonds = [("acetic_acid_C", "ethanol_O")]
    delete_atoms = ["acetic_acid_O", "acetic_acid_H", "ethanol_H"]
    rxn = AlchemicalReaction(
        select_dict=select_dict,
        create_bonds=create_bonds,
        delete_atoms=delete_atoms,
    )
    return rxn


class TestAlchemicalReaction:
    def test_smiles_to_universe(self):
        smiles = {"O": 2, "CCO": 2}
        universe = smiles_to_universe(smiles)
        assert len(universe.atoms) == 24
        assert len(universe.atoms.residues) == 4

    def test_build_half_reactions_w_atoms(self, select_dict):
        smiles = {"CC(=O)O": 2, "O": 2, "CCO": 2}
        create_bonds = [("acetic_acid_C", "ethanol_O")]
        delete_atoms = ["acetic_acid_O", "acetic_acid_H", "ethanol_H"]
        half_reactions, triggers_0, triggers_1 = AlchemicalReaction._build_half_reactions(
            smiles, select_dict, create_bonds, [], delete_atoms, return_trigger_atoms=True
        )
        # TODO: write a real test

    def test_build_half_reactions_w_bonds(self, select_dict):
        smiles = {"CC(=O)O": 2, "O": 2, "CCO": 2}
        create_bonds = [("acetic_acid_C", "ethanol_O"), ("acetic_acid_O", "ethanol_H")]
        delete_bonds = [("ethanol_O", "ethanol_H"), ("acetic_acid_O", "acetic_acid_C")]
        half_reactions, triggers_0, triggers_1 = AlchemicalReaction._build_half_reactions(
            smiles, select_dict, create_bonds, delete_bonds, [], return_trigger_atoms=True
        )
        # TODO: write a real test

    def test_get_reactive_atoms_df(self, acetic_ethanol_hydrolysis):
        smiles = {"CC(=O)O": 2, "O": 2, "CCO": 2}
        acetic_ethanol_hydrolysis.get_reactive_atoms_df(smiles)
        # TODO: write a real test
