from pymatgen.io.openmm.alchemy_utils import smiles_to_universe
import openff.toolkit as tk


class TestAlchemicalReaction:
    def test_smiles_to_universe(self):
        O = tk.Molecule.from_smiles("O")
        CCO = tk.Molecule.from_smiles("CCO")
        smiles = {O: 2, CCO: 2}
        universe = smiles_to_universe(smiles)
        assert len(universe.atoms) == 24
        assert len(universe.atoms.residues) == 4

    def test_get_reactive_system(self, acetic_ethanol_condensation):
        O = tk.Molecule.from_smiles("O")
        CCO = tk.Molecule.from_smiles("CCO")
        CCOO = tk.Molecule.from_smiles("CC(=O)O")
        openff_counts = {CCOO: 2, O: 2, CCO: 2}
        acetic_ethanol_condensation.make_reactive_system(openff_counts)
        return
        # TODO: write real tests
