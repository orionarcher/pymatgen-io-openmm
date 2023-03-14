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

    def test_make_acetic_atoms(self, acetic_rxn, acetic_rxn_rm_water):
        smile_dict = {"O": 2, "CCO": 5, "CC(=O)O": 2}  # water, ethanol, acetic
        openff_counts = {tk.Molecule.from_smiles(s): n for s, n in smile_dict.items()}

        # for testing purposes
        mda_atoms = openff_counts_to_universe(openff_counts).atoms

        reactive_atoms_1 = acetic_rxn.make_reactive_atoms(openff_counts)
        trig_left = reactive_atoms_1.trigger_atoms_left
        trig_right = reactive_atoms_1.trigger_atoms_right
        assert len(trig_left) == 2
        assert len(trig_right) == 5
        assert np.all(mda_atoms[trig_left].elements == "C")
        assert np.all(mda_atoms[trig_right].elements == "O")
        trigger_3 = reactive_atoms_1.trigger_atoms_right[3]
        half_reaction = reactive_atoms_1.half_reactions[trigger_3]
        assert len(half_reaction.delete_bonds) == 1
        assert len(half_reaction.create_bonds) == 2

        reactive_atoms_2 = acetic_rxn_rm_water.make_reactive_atoms(openff_counts)
        trig_left = reactive_atoms_2.trigger_atoms_left
        trig_right = reactive_atoms_2.trigger_atoms_right
        assert len(trig_left) == 2
        assert len(trig_right) == 5
        assert np.all(mda_atoms[trig_left].elements == "C")
        assert np.all(mda_atoms[trig_right].elements == "O")
        trigger = reactive_atoms_2.trigger_atoms_left[1]
        half_reaction = reactive_atoms_2.half_reactions[trigger]
        assert len(half_reaction.delete_atoms) == 2
        assert len(half_reaction.create_bonds) == 1
        # TODO: add rigorous testing

    def test_visualize_reactions(self, acetic_rxn, acetic_rxn_rm_water):
        smiles = ["O", "CCO", "CC(=O)O"]  # water, ethanol, acetic
        openff_mols = [tk.Molecule.from_smiles(s) for s in smiles]
        # acetic_rxn_rm_water.visualize_reactions(openff_mols)
        rdmol = acetic_rxn.visualize_reactions(openff_mols)

        assert len(rdmol.GetAtoms()) == sum(mol.n_atoms for mol in openff_mols)

    def test_make_propanedioic_system(self, propanedioic_rxn_rm_water):
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
