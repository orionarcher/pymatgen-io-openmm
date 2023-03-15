import copy

from pymatgen.io.openmm.alchemy_utils import (
    openff_counts_to_universe,
    HalfReaction,
    ReactiveAtoms,
    ReactiveSystem,
)
from pymatgen.io.openmm.utils import (
    molgraph_to_openff_topology,
)
import openff.toolkit as tk
import numpy as np


class TestAlchemicalReaction:
    def test_openff_counts_to_universe(self):
        smile_dict = {"O": 2, "CCO": 2}  # water, ethanol
        openff_counts = {tk.Molecule.from_smiles(s): n for s, n in smile_dict.items()}
        universe = openff_counts_to_universe(openff_counts)
        assert len(universe.atoms) == 24
        assert len(universe.atoms.residues) == 4
        res_lengths = [res.atoms.n_atoms for res in universe.atoms.residues]
        np.testing.assert_allclose(res_lengths, [3, 3, 9, 9])

    def test_make_acetic(self, acetic_rxn):
        smile_dict = {"O": 2, "CCO": 5, "CC(=O)O": 2}  # water, ethanol, acetic
        openff_counts = {tk.Molecule.from_smiles(s): n for s, n in smile_dict.items()}

        # for testing purposes
        mda_atoms = openff_counts_to_universe(openff_counts).atoms

        reactive_atoms = acetic_rxn.make_reactive_atoms(openff_counts)
        trig_left = reactive_atoms.trigger_atoms_left
        trig_right = reactive_atoms.trigger_atoms_right
        assert len(trig_left) == 2
        assert len(trig_right) == 5
        assert np.all(mda_atoms[trig_left].elements == "C")
        assert np.all(mda_atoms[trig_right].elements == "O")
        trigger_3 = reactive_atoms.trigger_atoms_right[3]
        half_reaction = reactive_atoms.half_reactions[trigger_3]
        assert len(half_reaction.delete_bonds) == 1
        assert len(half_reaction.create_bonds) == 2

    def test_make_acetic_del_water(self, acetic_rxn_rm_water):
        smile_dict = {"O": 2, "CCO": 5, "CC(=O)O": 2}  # water, ethanol, acetic
        openff_counts = {tk.Molecule.from_smiles(s): n for s, n in smile_dict.items()}

        # for testing purposes
        mda_atoms = openff_counts_to_universe(openff_counts).atoms

        reactive_atoms = acetic_rxn_rm_water.make_reactive_atoms(openff_counts)
        trig_left = reactive_atoms.trigger_atoms_left
        trig_right = reactive_atoms.trigger_atoms_right
        assert len(trig_left) == 2
        assert len(trig_right) == 5
        assert np.all(mda_atoms[trig_left].elements == "C")
        assert np.all(mda_atoms[trig_right].elements == "O")
        trigger = reactive_atoms.trigger_atoms_left[1]
        half_reaction = reactive_atoms.half_reactions[trigger]
        assert len(half_reaction.delete_atoms) == 2
        assert len(half_reaction.create_bonds) == 1

    def test_make_propanedioic_system(self, propanedioic_rxn_rm_water):
        smile_dict = {"O": 2, "CCO": 5, "C(C(=O)O)C(=O)O": 2}  # water, ethanol, acetic
        openff_counts = {tk.Molecule.from_smiles(s): n for s, n in smile_dict.items()}

        mda_atoms = openff_counts_to_universe(openff_counts).atoms

        reactive_atoms = propanedioic_rxn_rm_water.make_reactive_atoms(openff_counts)
        trig_left = reactive_atoms.trigger_atoms_left
        trig_right = reactive_atoms.trigger_atoms_right
        assert len(trig_left) == 4
        assert len(trig_right) == 5
        assert np.all(mda_atoms[trig_left].elements == "C")
        assert np.all(mda_atoms[trig_right].elements == "O")
        for trigger in reactive_atoms.trigger_atoms_left:
            half_reaction = reactive_atoms.half_reactions[trigger]
            assert len(half_reaction.delete_atoms) == 2
            assert len(half_reaction.create_bonds) == 1
        for trigger in reactive_atoms.trigger_atoms_right:
            half_reaction = reactive_atoms.half_reactions[trigger]
            assert len(half_reaction.delete_atoms) == 1
            assert len(half_reaction.create_bonds) == 1

    def test_visualize_reactions(
        self, acetic_rxn, acetic_rxn_rm_water, propanedioic_rxn_rm_water
    ):
        smiles = ["O", "CCO", "C(C(=O)O)C(=O)O"]  # water, ethanol, acetic
        openff_mols = [tk.Molecule.from_smiles(s) for s in smiles]
        # acetic_rxn_rm_water.visualize_reactions(openff_mols)
        rdmol = propanedioic_rxn_rm_water.visualize_reactions(openff_mols)

        assert len(rdmol.GetAtoms()) == sum(mol.n_atoms for mol in openff_mols)


class TestReactiveSystem:
    def test_from_reactions(self, acetic_rxn):
        smile_dict = {"O": 2, "CCO": 2, "CC(=O)O": 2}  # water, ethanol, acetic
        openff_counts = {tk.Molecule.from_smiles(s): n for s, n in smile_dict.items()}

        system = ReactiveSystem.from_reactions(openff_counts, [acetic_rxn])

        assert len(system.molgraph.molecule) == 40
        assert len(system.reactive_atom_sets) == 1

    def test_sample_reactions(self):
        half_reaction_l = HalfReaction(
            create_bonds=[1, 0], delete_atoms=[], delete_bonds=[]
        )
        half_reaction_r = HalfReaction(
            create_bonds=[2, 3], delete_atoms=[], delete_bonds=[]
        )
        reactive_atoms = ReactiveAtoms(
            half_reactions={1: half_reaction_l, 2: half_reaction_r},
            trigger_atoms_left=[1],
            trigger_atoms_right=[2],
            probability=0,
        )
        box_1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        reactions_1 = ReactiveSystem._sample_reactions(reactive_atoms, box_1, 1.5)
        assert len(reactions_1) == 0

        reactive_atoms.probability = 1
        reactions_2 = ReactiveSystem._sample_reactions(reactive_atoms, box_1, 1.5)
        assert len(reactions_2) == 1

        box_2 = np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0], [5, 0, 0]])
        reactions_2 = ReactiveSystem._sample_reactions(reactive_atoms, box_2, 1.5)
        assert len(reactions_2) == 0

    def test_react_molgraph_no_delete(self, acetic_rxn):
        smile_dict = {"O": 2, "CCO": 2, "CC(=O)O": 2}  # water, ethanol, acetic
        openff_counts = {tk.Molecule.from_smiles(s): n for s, n in smile_dict.items()}

        system = ReactiveSystem.from_reactions(openff_counts, [acetic_rxn])
        molgraph = copy.deepcopy(system.molgraph)

        old_to_new_map = {i: i for i in range(len(molgraph.molecule))}

        reactive_atoms = system.reactive_atom_sets[0]
        trig_index_l = reactive_atoms.trigger_atoms_left[0]
        trig_index_r = reactive_atoms.trigger_atoms_right[0]
        rxn_l = reactive_atoms.half_reactions[trig_index_l]
        rxn_r = reactive_atoms.half_reactions[trig_index_r]

        molgraph, old_to_new_map = ReactiveSystem._react_molgraph(
            molgraph, old_to_new_map, [(rxn_l, rxn_r)]
        )

        assert len(molgraph.graph.edges) == len(system.molgraph.graph.edges)
        assert len(molgraph) == len(system.molgraph)

        new_fragments = molgraph.get_disconnected_fragments()
        new_mol_sizes = [len(molgraph) for molgraph in new_fragments]
        assert sorted(new_mol_sizes) == [3, 3, 3, 8, 9, 14]

        topology_new = molgraph_to_openff_topology(molgraph)
        assert topology_new.n_unique_molecules == 4
        assert max(bond.bond_order for bond in topology_new.bonds) == 2

        topology_old = molgraph_to_openff_topology(system.molgraph)
        assert topology_old.n_unique_molecules == 3

    def test_react(self, acetic_rxn):
        smile_dict = {"O": 2, "CCO": 2, "CC(=O)O": 2}  # water, ethanol, acetic
        openff_counts = {tk.Molecule.from_smiles(s): n for s, n in smile_dict.items()}

        system = ReactiveSystem.from_reactions(openff_counts, [acetic_rxn])

        reactive_atoms = system.reactive_atom_sets[0]
        trig_index_l = reactive_atoms.trigger_atoms_left[0]
        trig_index_r = reactive_atoms.trigger_atoms_right[0]

        position_list = [[i * 3, 0, 0] for i in range(len(system.molgraph))]
        position_list[trig_index_l] = position_list[trig_index_r]
        positions = np.array(position_list)

        system.react(positions)

        molgraph = system.molgraph

        new_fragments = molgraph.get_disconnected_fragments()
        new_mol_sizes = [len(molgraph) for molgraph in new_fragments]
        assert sorted(new_mol_sizes) == [3, 3, 3, 8, 9, 14]

        topology_new = molgraph_to_openff_topology(molgraph)
        assert topology_new.n_unique_molecules == 4
        assert max(bond.bond_order for bond in topology_new.bonds) == 2

    def test_generate_topology(self):
        return


class TestReactiveAtoms:
    def test_remap(self):
        return


class TestHalfReaction:
    def test_remap(self):
        return
