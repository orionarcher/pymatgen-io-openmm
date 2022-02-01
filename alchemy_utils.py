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
        rxn_trigger_ix: int = 0,
    ):
        self.select_dict = select_dict if select_dict else {}
        self.create_bonds = create_bonds if create_bonds else []
        self.delete_bonds = delete_bonds if delete_bonds else []
        self.delete_atoms = delete_atoms if delete_atoms else []

    # TODO: need to make sure that we won't get an error if something reacts with itself
    @staticmethod
    def _extract_reactant_group(select_dict, reactant_list, universe, return_ix):  # rename
        atom_list = []
        for reactant_names in reactant_list:
            if isinstance(reactant_names, str):
                select_str = select_dict[reactant_names]
                atoms = universe.select_atoms(select_str)
                atoms = atoms.ix if return_ix else atoms
                atom_list.append(atoms)
            elif isinstance(reactant_names, tuple):
                select_strs = [select_dict[reactant_name] for reactant_name in reactant_names]
                atoms_tuple = tuple(universe.select_atoms(select_str) for select_str in select_strs)
                atoms_tuple = tuple(atom_group.ix for atom_group in atoms_tuple) if return_ix else atoms_tuple
                atom_list.append(atoms_tuple)
            else:
                Exception("reactant_list must be a list of tuple or str")
        return atom_list

    @staticmethod
    def _build_reactive_atoms_df(universe, select_dict, create_bonds, delete_bonds, delete_atoms):
        participating_atoms = []
        for rxn_type, bonds in {"create_bonds": create_bonds, "delete_bonds": delete_bonds}.items():
            for bond_n, bond in enumerate(bonds):
                atoms_ix_0 = universe.select_atoms(select_dict[bond[0]]).ix
                atoms_ix_1 = universe.select_atoms(select_dict[bond[1]]).ix
                for atom_ix_list in [atoms_ix_0, atoms_ix_1]:
                    for atom_ix in atom_ix_list:
                        res_ix = universe.atoms[atom_ix].residue.ix
                        participating_atoms.append(
                            {"atom_ix": atom_ix, "res_ix": res_ix, "type": rxn_type, "bond_n": bond_n}
                        )
        for atoms in delete_atoms:
            atom_ix_array = universe.select_atoms(select_dict[atoms]).ix
            for atom_ix in atom_ix_array:
                participating_atoms.append({"atom_ix": atom_ix, "type": "delete_atom", "bond_n": None})
        return pd.DataFrame(participating_atoms)

    @staticmethod
    def _build_half_reactions_dict(df, universe):
        trigger_atom_ix = df[((df.type == "create_bonds") & (df.bond_n == 0))]["atom_ix"]
        half_reactions = {}
        for ix in trigger_atom_ix:
            within_two_bonds = f"(index {ix}) or (bonded index {ix}) or (bonded bonded index {ix})"
            nearby_atoms_ix = universe.select_atoms(within_two_bonds).ix
            atoms_df = df[np.isin(df["atom_ix"], nearby_atoms_ix)]
            create_ix = list(atoms_df[atoms_df.type == "create_bonds"].sort_values("bond_n")["atom_ix"])
            delete_ix = atoms_df[atoms_df.type == "delete_bonds"]
            unique_bond_n = delete_ix["bond_n"].unique()
            delete_ix = [tuple(delete_ix[delete_ix["bond_n"] == bond_n]["atom_ix"].values) for bond_n in unique_bond_n]
            delete_atom_ix = list(atoms_df[atoms_df.type == "delete_atom"]["atom_ix"].values)
            half_reaction = {"create_bonds": create_ix, "delete_bonds": delete_ix, "delete_atoms": delete_atom_ix}
            half_reactions[ix] = half_reaction
        return half_reactions

    @staticmethod
    def _build_half_reactions(smiles, select_dict, create_bonds, delete_bonds, delete_atoms):
        smiles_1 = {smile: 1 for smile in smiles.keys()}
        universe = smiles_to_universe(smiles_1)
        df = AlchemicalReaction._build_reactive_atoms_df(
            universe, select_dict, create_bonds, delete_bonds, delete_atoms
        )
        half_reactions = AlchemicalReaction._build_half_reactions_dict(df, universe)

        return half_reactions
        # TODO: loop to create_bonds many half_reactions

    # def get_bonds_to_create(self, smiles, return_ix=False):
    #     universe = get_openmm_topology(smiles)
    #     return AlchemicalReaction._extract_reactant_group(self.select_dict, self.create_bonds, universe, return_ix)
    #
    # def get_bonds_to_delete(self, smiles, return_ix=False):
    #     universe = get_openmm_topology(smiles)
    #     return AlchemicalReaction._extract_reactant_group(self.select_dict, self.delete_bonds, universe, return_ix)
    #
    # def get_atoms_to_delete(self, smiles, return_ix=False):
    #     universe = get_openmm_topology(smiles)
    #     return AlchemicalReaction._extract_reactant_group(self.select_dict, self.delete_atoms, universe, return_ix)
