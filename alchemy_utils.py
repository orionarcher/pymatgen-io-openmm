"""
Utilities for implementing AlchemicalReactions
"""

from typing import List, Tuple, Dict
from io import StringIO
import pandas as pd

from monty.json import MSONable

import numpy as np
import MDAnalysis as mda

from pymatgen.io.openmm.inputs import TopologyInput
from pymatgen.io.openmm.utils import get_openmm_topology


def smiles_to_universe(smiles):
    """
    Quick conversion from a set of smiles to a MDanalysis Universe.

    Args:
        smiles: a dict of smiles: counts.

    Returns:
        A MDanalysis universe

    """
    topology_input = TopologyInput(get_openmm_topology(smiles))
    with StringIO(topology_input.get_string()) as topology_file:
        universe = mda.Universe(topology_file, format="pdb")
    return universe


class AlchemicalReaction(MSONable):
    """
    An AlchemicalReaction for use in a OpenmmAlchemyGen generator.
    """

    def __init__(
        self,
        select_dict: Dict[str, str] = None,
        create_bonds: List[Tuple[str, str]] = None,
        delete_bonds: List[Tuple[str, str]] = None,
        delete_atoms: List[str] = None,
    ):
        self.select_dict = select_dict if select_dict else {}
        self.create_bonds = create_bonds if create_bonds else []
        self.delete_bonds = delete_bonds if delete_bonds else []
        self.delete_atoms = delete_atoms if delete_atoms else []

    # TODO: need to make sure that we won't get an error if something reacts with itself

    @staticmethod
    def _build_reactive_atoms_df(universe, select_dict, create_bonds, delete_bonds, delete_atoms):
        participating_atoms = []
        # loop to reuse code for create_bonds and delete_bonds
        for rxn_type, bonds in {
            "create_bonds": create_bonds,
            "delete_bonds": delete_bonds,
        }.items():
            # loop through every unique bond type
            for bond_n, bond in enumerate(bonds):
                atoms_ix_0 = universe.select_atoms(select_dict[bond[0]]).ix
                atoms_ix_1 = universe.select_atoms(select_dict[bond[1]]).ix
                # loop through each type of atom in bond
                for half_rxn_ix, atom_ix_list in enumerate([atoms_ix_0, atoms_ix_1]):
                    # loop through each unique atom of that type
                    for atom_ix in atom_ix_list:
                        res_ix = universe.atoms[atom_ix].residue.ix
                        participating_atoms.append(
                            {
                                "atom_ix": atom_ix,
                                "res_ix": res_ix,
                                "type": rxn_type,
                                "bond_n": bond_n,
                                "half_rxn_ix": half_rxn_ix,
                            }
                        )
        # loop through atoms of each type
        for atoms in delete_atoms:
            atom_ix_array = universe.select_atoms(select_dict[atoms]).ix
            # loop through each unique atom of that type
            for atom_ix in atom_ix_array:
                res_ix = universe.atoms[atom_ix].residue.ix
                participating_atoms.append(
                    {
                        "atom_ix": atom_ix,
                        "res_ix": res_ix,
                        "type": "delete_atom",
                        "bond_n": None,
                    }
                )
        return pd.DataFrame(participating_atoms)

    @staticmethod
    def _get_trigger_atoms(df, universe):
        trigger_atom_ix = df[((df.type == "create_bonds") & (df.bond_n == 0))]["atom_ix"]
        trigger_atom_dfs = []
        for ix in trigger_atom_ix:
            within_two_bonds = f"(index {ix}) or (bonded index {ix}) or (bonded bonded index {ix})"
            nearby_atoms_ix = universe.select_atoms(within_two_bonds).ix
            atoms_df = df[np.isin(df["atom_ix"], nearby_atoms_ix)]
            atoms_df["trigger_ix"] = ix
            trigger_atom_dfs.append(atoms_df)
        return pd.concat(trigger_atom_dfs)

    @staticmethod
    def _expand_to_all_atoms(trig_df, res_sizes, res_counts):
        trig_df = trig_df.sort_values("res_ix")
        res_offsets = np.cumsum(np.array(res_sizes) * (np.array(res_counts) - 1))
        res_offsets = np.insert(res_offsets, 0, 0)
        big_df_list = []
        for res_ix, res_df in trig_df.groupby(["res_ix"]):
            expanded_df = pd.concat([res_df] * res_counts[res_ix])
            n_atoms = len(res_df)
            offsets = np.arange(res_offsets[res_ix], res_offsets[res_ix + 1] + 1, res_sizes[res_ix])
            offset_array = np.repeat(offsets, n_atoms)
            expanded_df.atom_ix += offset_array
            expanded_df.trigger_ix += offset_array
            big_df_list += [expanded_df]
        big_df = pd.concat(big_df_list)
        return big_df

    @staticmethod
    def _build_half_reactions_dict(all_atoms_df):
        half_reactions = {}
        for trigger_ix, atoms_df in all_atoms_df.groupby(["trigger_ix"]):
            create_ix = list(atoms_df[atoms_df.type == "create_bonds"].sort_values("bond_n")["atom_ix"])
            delete_ix = atoms_df[atoms_df.type == "delete_bonds"]
            unique_bond_n = delete_ix["bond_n"].unique()
            delete_ix = [tuple(delete_ix[delete_ix["bond_n"] == bond_n]["atom_ix"].values) for bond_n in unique_bond_n]
            delete_atom_ix = list(atoms_df[atoms_df.type == "delete_atom"]["atom_ix"].values)
            half_reaction = {
                "create_bonds": create_ix,
                "delete_bonds": delete_ix,
                "delete_atoms": delete_atom_ix,
            }
            half_reactions[trigger_ix] = half_reaction
        return half_reactions

    @staticmethod
    def _build_half_reactions(
        smiles,
        select_dict,
        create_bonds,
        delete_bonds,
        delete_atoms,
        return_trigger_atoms=False,
        return_atoms_df=False,
    ):
        smiles_1 = {smile: 1 for smile in smiles.keys()}
        universe = smiles_to_universe(smiles_1)
        df = AlchemicalReaction._build_reactive_atoms_df(
            universe, select_dict, create_bonds, delete_bonds, delete_atoms
        )
        trig_df = AlchemicalReaction._get_trigger_atoms(df, universe)
        res_sizes = [res.atoms.n_atoms for res in universe.residues]
        res_counts = list(smiles.values())
        all_atoms_df = AlchemicalReaction._expand_to_all_atoms(trig_df, res_sizes, res_counts)
        trigger_atoms_0 = all_atoms_df[
            (all_atoms_df.type == "create_bonds") & (all_atoms_df.bond_n == 0) & (all_atoms_df.half_rxn_ix == 0)
        ]["trigger_ix"].values
        trigger_atoms_1 = all_atoms_df[
            (all_atoms_df.type == "create_bonds") & (all_atoms_df.bond_n == 0) & (all_atoms_df.half_rxn_ix == 1)
        ]["trigger_ix"].values
        half_reactions = AlchemicalReaction._build_half_reactions_dict(all_atoms_df)
        return_args = tuple([half_reactions])
        if return_trigger_atoms:
            return_args += (trigger_atoms_0, trigger_atoms_1)
        if return_atoms_df:
            return_args += tuple([all_atoms_df])
        return return_args

    def get_half_reactions_and_trigger_atoms(self, smiles):
        """

        Args:
            smiles:

        Returns:
            half_reactions, trigger_atoms_0, trigger_atoms_1
        """
        return AlchemicalReaction._build_half_reactions(
            smiles,
            self.select_dict,
            self.create_bonds,
            self.delete_bonds,
            self.delete_atoms,
            return_trigger_atoms=True,
        )

    def get_reactive_atoms_df(self, smiles):
        """
        Returns the DataFrame of all atoms that participate in the reaction. Intended primarily for debugging.

        Args:
            smiles:

        Returns:

        """
        _, atoms_df = AlchemicalReaction._build_half_reactions(
            smiles,
            self.select_dict,
            self.create_bonds,
            self.delete_bonds,
            self.delete_atoms,
            return_atoms_df=True,
        )
        return atoms_df[["atom_ix", "type", "trigger_ix"]]

    @staticmethod
    def get_universe(smiles):
        """
        This is a wrapper around utils.smiles_to_universe.

        Args:
            smiles:

        Returns:

        """
        return smiles_to_universe(smiles)
