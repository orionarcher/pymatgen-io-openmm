import MDAnalysis as mda

import numpy as np
import numpy.testing
import pytest

from pymatgen.io.openmm.alchemy_utils import AlchemicalReaction, smiles_to_universe
from pymatgen.io.openmm.tests.datafiles import acetic_ethanol_pdb


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
        assert len(universe.atoms) == 12
        assert len(universe.atoms.residues) == 2

    def test_extract_reactant_group_w_atoms(self, select_dict):
        smiles = {"CC(=O)O": 2, "O": 3, "CCO": 2}
        create_bonds = [("acetic_acid_C", "ethanol_O")]
        delete_atoms = ["acetic_acid_O", "acetic_acid_H", "ethanol_H"]
        AlchemicalReaction._build_half_reactions(smiles, select_dict, create_bonds, [], delete_atoms)

    def test_extract_reactant_group_w_bonds(self, select_dict):
        smiles = {"CC(=O)O": 2, "O": 3, "CCO": 2}
        create_bonds = [("acetic_acid_C", "ethanol_O"), ("acetic_acid_O", "ethanol_H")]
        delete_bonds = [("ethanol_O", "ethanol_H"), ("acetic_acid_O", "acetic_acid_C")]
        AlchemicalReaction._build_half_reactions(smiles, select_dict, create_bonds, delete_bonds, [])

    def test_extract_reactant_group_funcs(
        self,
        acetic_ethanol_hydrolysis_del_water,
        acetic_ethanol_hydrolysis,
    ):
        universe = mda.Universe(acetic_ethanol_pdb, format="pdb")

        rxn1 = acetic_ethanol_hydrolysis_del_water
        bonds = rxn1.get_bonds_to_create(universe)
        assert len(bonds[0][0]) == 20
        assert len(bonds[0][1]) == 20
        atoms = rxn1.get_atoms_to_delete(universe)
        n_atoms = [len(atom_group) for atom_group in atoms]
        np.testing.assert_equal(n_atoms, [20, 20, 20])

        rxn2 = acetic_ethanol_hydrolysis
        c_bonds = rxn2.get_bonds_to_create(universe)
        n_c_bonds = [[len(c_bonds[i][j]) for i in (0, 1)] for j in (0, 1)]
        numpy.testing.assert_allclose(n_c_bonds, 20)
        d_bonds = rxn2.get_bonds_to_delete(universe)
        n_d_bonds = [[len(d_bonds[i][j]) for i in (0, 1)] for j in (0, 1)]
        numpy.testing.assert_allclose(n_d_bonds, 20)
