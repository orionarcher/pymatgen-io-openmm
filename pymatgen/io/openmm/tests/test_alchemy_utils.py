from pymatgen.io.openmm.alchemy_utils import openff_counts_to_universe
import openff.toolkit as tk
import numpy as np


class TestAlchemicalReaction:
    def test_openff_counts_to_universe(self):
        smile_dict = {"O": 2, "CCO": 2}  # water, ethanol, acetic
        openff_counts = {tk.Molecule.from_smiles(s): n for s, n in smile_dict.items()}
        universe = openff_counts_to_universe(openff_counts)
        assert len(universe.atoms) == 24
        assert len(universe.atoms.residues) == 4
        res_lengths = [res.atoms.n_atoms for res in universe.atoms.residues]
        np.testing.assert_allclose(res_lengths, [3, 3, 9, 9])

    def test_make_acetic_system(self, acetic_rxn, acetic_rxn_rm_water):
        smile_dict = {"O": 2, "CCO": 2, "CC(=O)O": 2}  # water, ethanol, acetic
        openff_counts = {tk.Molecule.from_smiles(s): n for s, n in smile_dict.items()}

        acetic_rxn.make_reactive_atoms(openff_counts)

        acetic_rxn_rm_water.make_reactive_atoms(openff_counts)
        # TODO: add rigorous testing

    def test_visualize_reactions(self, acetic_rxn, acetic_rxn_rm_water):
        smiles = ["O", "CCO", "CC(=O)O"]  # water, ethanol, acetic
        openff_mols = [tk.Molecule.from_smiles(s) for s in smiles]
        # acetic_rxn_rm_water.visualize_reactions(openff_mols)
        rdmol = acetic_rxn.visualize_reactions(openff_mols)

        assert len(rdmol.GetAtoms()) == sum(mol.n_atoms for mol in openff_mols)

    def test_make_propanedioic_system(self, mols_dict, propanedioic_rxn_rm_water):
        # water, ethanol, propanedioic
        smiles = ["O", "CCO", "C(C(=O)O)C(=O)O"]  # water, ethanol, propanedioic
        openff_mols = [tk.Molecule.from_smiles(s) for s in smiles]
        propanedioic_rxn_rm_water.visualize_reactions(openff_mols)
        # TODO: add rigorous testing


class TestReactiveSystem:
    def test_from_reactions(self):
        return

    def test_sample_reactions(self):
        return

    def test_react_molgraph(self):
        return

    def test_react(self):
        return

    def test_generate_topology(self):
        return


class TestReactiveAtoms:
    def test_remap(self):
        return


class TestHalfReaction:
    def test_remap(self):
        return
