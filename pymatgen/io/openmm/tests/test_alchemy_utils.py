from pymatgen.io.openmm.alchemy_utils import openff_counts_to_universe
import openff.toolkit as tk


class TestAlchemicalReaction:
    def test_smiles_to_universe(self):
        O = tk.Molecule.from_smiles("O")
        CCO = tk.Molecule.from_smiles("CCO")
        smiles = {O: 2, CCO: 2}
        universe = openff_counts_to_universe(smiles)
        assert len(universe.atoms) == 24
        assert len(universe.atoms.residues) == 4

    def test_get_reactive_system(
        self, acetic_ethanol_condensation, acetic_ethanol_condensation_del_water
    ):
        O = tk.Molecule.from_smiles("O")
        CCO = tk.Molecule.from_smiles("CCO")
        CCOO = tk.Molecule.from_smiles("CC(=O)O")
        openff_counts = {CCOO: 2, O: 2, CCO: 2}

        # acetic_ethanol_condensation.make_reactive_atoms(openff_counts)

        acetic_ethanol_condensation_del_water.make_reactive_atoms(openff_counts)
        return
        # TODO: add rigorous testing

    def test_visualize_reactions(
        self, acetic_ethanol_condensation, acetic_ethanol_condensation_del_water
    ):
        O = tk.Molecule.from_smiles("O")
        CCO = tk.Molecule.from_smiles("CCO")
        CCOO = tk.Molecule.from_smiles("CC(=O)O")
        openff_mols = [O, CCO, CCOO]
        # acetic_ethanol_condensation_del_water.visualize_reactions(openff_mols)
        acetic_ethanol_condensation.visualize_reactions(openff_mols)

    def test_propanedioic(self, propanedioic_ethanol_condensation_del_water):
        O = tk.Molecule.from_smiles("O")
        CCO = tk.Molecule.from_smiles("CCO")
        CCOO = tk.Molecule.from_smiles("C(C(=O)O)C(=O)O")
        openff_mols = [O, CCO, CCOO]
        propanedioic_ethanol_condensation_del_water.visualize_reactions(
            openff_mols,
        )
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
