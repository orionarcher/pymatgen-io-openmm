"""
Concrete implementations of InputSet for the OpenMM IO.
"""

# base python
import json
from pathlib import Path
from typing import Union, Optional, Dict

from monty.json import MontyDecoder

# openmm
import openmm
from openmm.openmm import Platform
import openff
from openmm.app import Simulation, PME
from openmm.app.modeller import Modeller
import numpy as np

from MDAnalysis.lib.distances import capped_distance


# pymatgen
from pymatgen.io.core import InputSet
from pymatgen.io.openmm.inputs import (
    TopologyInput,
    SystemInput,
    IntegratorInput,
    StateInput,
)

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"

from pymatgen.io.openmm.utils import smiles_in_topology, assign_small_molecule_ff


class OpenMMSet(InputSet):
    """
    An InputSet for OpenMM Simulations.

    This is a container for the serialized topology, system, integrator, and state files.
    Topology is stored as a pdb and the other files are stored as xml. The state file is
    optional but positions must be manually assigned if no state is provided.
    """

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        topology_file: str = "topology.pdb",
        system_file: str = "system.xml",
        integrator_file: str = "integrator.xml",
        state_file: str = "state.xml",
    ):
        """
        Instantiates an InputSet from a directory containing a Topology PDB, a
        System XML, an Integrator XML, and (optionally) a State XML. If no State
        is given, system coordinates must be manually assigned.

        Args:
            directory: directory that holds the input files.
            topology_file: name of the pdb file with topological information.
            system_file: name of the serialized System xml file.
            integrator_file: name of the serialized Integrator xml file.
            state_file: name of the serialized State xml file. If there is no state_file,
                then positions must be set for simulation.

        Returns:
            an OpenMMSet
        """
        # TODO: will need to add some sort of settings file
        source_dir = Path(directory)
        topology_input = TopologyInput.from_file(source_dir / topology_file)
        system_input = SystemInput.from_file(source_dir / system_file)
        integrator_input = IntegratorInput.from_file(source_dir / integrator_file)
        inputs = {
            topology_file: topology_input,
            system_file: system_input,
            integrator_file: integrator_input,
        }
        if Path(source_dir / state_file).is_file():
            openmm_set = OpenMMSet(
                inputs=inputs,  # type: ignore
                topology_file=topology_file,
                system_file=system_file,
                integrator_file=integrator_file,
                state_file=state_file,
            )
            openmm_set.inputs[state_file] = StateInput.from_file(
                source_dir / state_file
            )
        else:
            openmm_set = OpenMMSet(
                inputs=inputs,  # type: ignore
                topology_file=topology_file,
                system_file=system_file,
                integrator_file=integrator_file,
            )
        return openmm_set

    def validate(self) -> bool:
        """
        Checks that the OpenMMSet will generate a valid simulation.

        Returns:
            bool
        """
        try:
            self.get_simulation()
        except Exception as e:
            print(
                "A valid simulation could not be generated, the following error was raised:",
                e,
            )
            return False
        else:
            return True

    def get_simulation(
        self,
        platform: Optional[Union[str, openmm.openmm.Platform]] = None,  # type: ignore
        platformProperties: Optional[Dict[str, str]] = None,
    ) -> Simulation:
        """
        Instantiates and returns an OpenMM.Simulation from the input files.

        Args:
            platform: the OpenMM platform passed to the Simulation.
            platformProperties: properties of the OpenMM platform that is passed to the simulation.

        Returns:
            OpenMM.Simulation
        """
        topology_input = self.inputs[self.topology_file]
        system_input = self.inputs[self.system_file]
        integrator_input = self.inputs[self.integrator_file]
        if isinstance(platform, str):
            platform = openmm.openmm.Platform.getPlatformByName(platform)  # type: ignore
        simulation = Simulation(
            topology_input.get_topology(),  # type: ignore
            system_input.get_system(),  # type: ignore
            integrator_input.get_integrator(),  # type: ignore
            platform=platform,
            platformProperties=platformProperties,
        )
        if hasattr(self, "state_file") and self.state_file:
            # TODO: confirm that this works correctly
            state_input = self.inputs[self.state_file]
            simulation.context.setState(state_input.get_state())  # type: ignore
        return simulation


class OpenMMAlchemySet(OpenMMSet):
    """
    An InputSet for an alchemical reaction workflow.
    """

    @classmethod
    def from_directory(cls, directory: Union[str, Path], rxn_atoms_file="reaction_spec.json", **kwargs):  # type: ignore
        input_set = super().from_directory(directory, **kwargs)
        source_dir = Path(directory)
        with open(source_dir / rxn_atoms_file) as file:
            file_string = file.read()
        input_set.inputs = {
            **input_set.inputs,
            rxn_atoms_file: file_string,
        }
        input_set.rxn_atoms_file = rxn_atoms_file
        input_set.__class__ = OpenMMAlchemySet
        return input_set

    def run(
        self,
        n_cycles,
        steps_per_cycle,
        initial_steps: int = 0,
        cutoff_distance: float = 4,
        platform: Optional[Union[str, Platform]] = None,
        platformProperties: Optional[Dict[str, str]] = None,
    ):
        """
        Run the networking procedure for n_cycles.

        Args:
            cycles:

        Returns:

        """
        self._prepare()
        topology = self.inputs[self.topology_file].get_topology()
        system = self.inputs[self.system_file].get_system()
        integrator_input = self.inputs[self.integrator_file]
        if isinstance(platform, str):
            platform = Platform.getPlatformByName(platform)
        assert (
            hasattr(self, "state_file") and self.state_file
        ), "an AlchemySet must have a state to run"
        state = self.inputs[self.state_file].get_state()
        for i in range(n_cycles):
            integrator = integrator_input.get_integrator()
            simulation = Simulation(
                topology,  # type: ignore
                system,  # type: ignore
                integrator,  # type: ignore
                platform=platform,
                platformProperties=platformProperties,
            )
            simulation.context.setState(state)  # type: ignore
            if i == 0:
                simulation.minimizeEnergy()
                simulation.step(initial_steps)
                simulation.minimizeEnergy()

            else:
                simulation.minimizeEnergy()
                simulation.step(steps_per_cycle)
                simulation.minimizeEnergy()

            state = simulation.context.getState(
                getPositions=True,
                getVelocities=True,
            )
            positions = state.getPositions(asNumpy=True)._value
            topology = TopologyInput(topology, positions).get_topology()
            topology = self._update_topology(topology, positions, cutoff_distance)
            smiles = smiles_in_topology(topology, positions)
            openff_mols = [
                openff.toolkit.topology.Molecule.from_smiles(smile) for smile in smiles
            ]
            ff_template = assign_small_molecule_ff(openff_mols, "sage")
            # quick and dirty re-parameterization
            # TODO: perhaps instead of using the topology to add bonds with the modeler
            #  we can instead maintain a molecule list (in OpenBabel?) and add and remove
            #  bonds from that, allowing us to keep the topology information consistent.
            # then generate new
            # another

            # TODO: reparameterize with GAFF to avoid smiles!
            forcefield = openmm.app.ForceField()
            forcefield.registerTemplateGenerator(ff_template.generator)
            nonbondedCutoff = 5  # TODO: change this! use state!
            system = forcefield.createSystem(
                topology=topology, nonbondedMethod=PME, nonbondedCutoff=nonbondedCutoff
            )
            # openff_forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
            # openff_topology = openff.toolkit.topology.Topology.from_openmm(
            #     topology,
            #     unique_molecules=openff_mols,
            # )
            # # openff_topology.box_vectors = box_vectors
            # system = openff_forcefield.create_openmm_system(
            #     openff_topology,
            #     allow_nonintegral_charges=True,
            # )
            system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
            print("loop")

        self._conclude()
        return 1

    def _prepare(self):
        rxn_spec = json.loads(self.inputs[self.rxn_atoms_file], cls=MontyDecoder)
        self.half_reactions = {
            int(i): spec for i, spec in rxn_spec["half_reactions"].items()
        }
        self.trigger_atoms = rxn_spec["trigger_atoms"]
        self.index_map = np.array(rxn_spec["current_to_original_index"])
        self.force_field = rxn_spec["force_field"]

    def _conclude(self):
        """
        Conclude and update files
        """
        self._update_input_files()
        del self.half_reactions
        del self.trigger_atoms
        del self.index_map
        del self.force_field

    @staticmethod
    def _create_bonds(topology, topology_atoms, reactions):
        """
        This should add pairs of half reactions directly to the provided topology.

        Args:
            topology: an OpenMM topology
            topology_atoms: a dictionary of atoms in the topology, keys are ids, values are Atoms
            reactions: tuples of half_reaction dicts

        Returns:

        """
        for half_reaction_0, half_reaction_1 in reactions:
            # reindex rxn_atom_0 to index in topology
            # create bonds
            bonds_to_make = zip(
                half_reaction_0["create_bonds"], half_reaction_1["create_bonds"]
            )
            # in theory, this is simple
            for ix_0, ix_1 in bonds_to_make:
                atom_0, atom_1 = topology_atoms[ix_0], topology_atoms[ix_1]
                topology.addBond(atom_0, atom_1)
        return topology

    @staticmethod
    def _delete_bonds_and_atoms(topology, topology_atoms, reactions, positions):
        """
        Creates a Modeller object and uses it to delete atoms and bonds in the half reactions.

        Args:
            topology: an OpenMM topology
            topology_atoms: a dictionary of atoms in the topology, keys are ids, values are Atoms
            reactions: tuples of half_reaction dicts
            positions: the physical positions of each atom, needed to instantiate a Modeler

        Returns:

        """
        modeller = Modeller(topology, positions)
        bonds_to_delete = []
        atoms_to_delete = []
        for half_reaction_0, half_reaction_1 in reactions:
            # generate bonds to delete
            bonds_to_delete_ix = (
                half_reaction_0["delete_bonds"] + half_reaction_1["delete_bonds"]
            )
            for ix_0, ix_1 in bonds_to_delete_ix:
                atom_0, atom_1 = topology_atoms[ix_0], topology_atoms[ix_1]
                bonds_to_delete.append((atom_0, atom_1))
            # generate atoms to delete
            atoms_to_delete_ix = (
                half_reaction_0["delete_atoms"] + half_reaction_1["delete_atoms"]
            )
            for ix_0, ix_1 in atoms_to_delete_ix:
                atom_0, atom_1 = topology_atoms[ix_0], topology_atoms[ix_1]
                atoms_to_delete.append((atom_0, atom_1))
        # delete bonds and atoms
        modeller.delete(bonds_to_delete + atoms_to_delete)
        return modeller.getTopology()

    def _update_topology(self, topology, positions, cutoff_distance):
        """
        A single cycle of the reaction scheme.

        Args:
            topology: an OpenMM topology
            positions: the positions of each atom in the topology.
            cutoff_distance: cutoff distance to be used on the trigger atoms
        """
        topology_atoms = {
            self.index_map[i]: atom for i, atom in enumerate(topology.atoms())
        }
        atoms_ix_0 = self.index_map[self.trigger_atoms[0]]
        atoms_ix_1 = self.index_map[self.trigger_atoms[1]]
        # TODO: will return indices in atoms_0 array, not in topology. Need to reindex in loop.
        positions_0, positions_1 = positions[atoms_ix_0], positions[atoms_ix_1]
        reaction_pairs = capped_distance(
            positions_0, positions_1, cutoff_distance, return_distances=False
        )
        reactions = []
        for ix_0, ix_1 in reaction_pairs:
            half_reaction_0 = self.half_reactions[atoms_ix_0[ix_0]]
            half_reaction_1 = self.half_reactions[atoms_ix_1[ix_1]]
            reactions.append((half_reaction_0, half_reaction_1))
        # TODO: we never remove the atoms from self.trigger_atoms!!!!!!
        topology = self._create_bonds(topology, topology_atoms, reactions)
        topology = self._delete_bonds_and_atoms(
            topology, topology_atoms, reactions, positions
        )
        return topology

    def _update_input_files(self):
        """
        Internal method to update the topology and system files

        Returns:

        """


class OldOpenMMAlchemySet(OpenMMSet):
    """
    An InputSet for an alchemical reaction workflow.
    """

    @classmethod
    def from_directory(cls, directory: Union[str, Path], rxn_atoms_file="reaction_spec.json", **kwargs):  # type: ignore
        input_set = super().from_directory(directory, **kwargs)
        source_dir = Path(directory)
        with open(source_dir / rxn_atoms_file) as file:
            file_string = file.read()
        input_set.inputs = {
            **input_set.inputs,
            rxn_atoms_file: file_string,
        }
        input_set.rxn_atoms_file = rxn_atoms_file
        input_set.__class__ = OpenMMAlchemySet
        return input_set

    def run(
        self,
        n_cycles,
        steps_per_cycle,
        initial_steps: int = 0,
        cutoff_distance: float = 4,
        platform: Optional[Union[str, Platform]] = None,
        platformProperties: Optional[Dict[str, str]] = None,
    ):
        """
        Run the networking procedure for n_cycles.

        Args:
            cycles:

        Returns:

        """
        self._prepare()
        topology = self.inputs[self.topology_file].get_topology()
        system = self.inputs[self.system_file].get_system()
        integrator_input = self.inputs[self.integrator_file]
        if isinstance(platform, str):
            platform = Platform.getPlatformByName(platform)
        assert (
            hasattr(self, "state_file") and self.state_file
        ), "an AlchemySet must have a state to run"
        state = self.inputs[self.state_file].get_state()
        for i in range(n_cycles):
            integrator = integrator_input.get_integrator()
            simulation = Simulation(
                topology,  # type: ignore
                system,  # type: ignore
                integrator,  # type: ignore
                platform=platform,
                platformProperties=platformProperties,
            )
            simulation.context.setState(state)  # type: ignore
            if i == 0:
                simulation.minimizeEnergy()
                simulation.step(initial_steps)
                simulation.minimizeEnergy()

            else:
                simulation.minimizeEnergy()
                simulation.step(steps_per_cycle)
                simulation.minimizeEnergy()

            state = simulation.context.getState(
                getPositions=True,
                getVelocities=True,
            )
            positions = state.getPositions(asNumpy=True)._value
            topology = TopologyInput(topology, positions).get_topology()
            topology = self._update_topology(topology, positions, cutoff_distance)
            smiles = smiles_in_topology(topology, positions)
            openff_mols = [
                openff.toolkit.topology.Molecule.from_smiles(smile) for smile in smiles
            ]
            ff_template = assign_small_molecule_ff(openff_mols, "sage")
            # quick and dirty re-parameterization
            # TODO: perhaps instead of using the topology to add bonds with the modeler
            #  we can instead maintain a molecule list (in OpenBabel?) and add and remove
            #  bonds from that, allowing us to keep the topology information consistent.
            # then generate new
            # another

            # TODO: reparameterize with GAFF to avoid smiles!
            forcefield = openmm.app.ForceField()
            forcefield.registerTemplateGenerator(ff_template.generator)
            nonbondedCutoff = 5  # TODO: change this! use state!
            system = forcefield.createSystem(
                topology=topology, nonbondedMethod=PME, nonbondedCutoff=nonbondedCutoff
            )
            # openff_forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
            # openff_topology = openff.toolkit.topology.Topology.from_openmm(
            #     topology,
            #     unique_molecules=openff_mols,
            # )
            # # openff_topology.box_vectors = box_vectors
            # system = openff_forcefield.create_openmm_system(
            #     openff_topology,
            #     allow_nonintegral_charges=True,
            # )
            system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
            print("loop")

        self._conclude()
        return 1

    def _prepare(self):
        rxn_spec = json.loads(self.inputs[self.rxn_atoms_file], cls=MontyDecoder)
        self.half_reactions = {
            int(i): spec for i, spec in rxn_spec["half_reactions"].items()
        }
        self.trigger_atoms = rxn_spec["trigger_atoms"]
        self.index_map = np.array(rxn_spec["current_to_original_index"])
        self.force_field = rxn_spec["force_field"]

    def _conclude(self):
        """
        Conclude and update files
        """
        self._update_input_files()
        del self.half_reactions
        del self.trigger_atoms
        del self.index_map
        del self.force_field

    @staticmethod
    def _create_bonds(topology, topology_atoms, reactions):
        """
        This should add pairs of half reactions directly to the provided topology.

        Args:
            topology: an OpenMM topology
            topology_atoms: a dictionary of atoms in the topology, keys are ids, values are Atoms
            reactions: tuples of half_reaction dicts

        Returns:

        """
        for half_reaction_0, half_reaction_1 in reactions:
            # reindex rxn_atom_0 to index in topology
            # create bonds
            bonds_to_make = zip(
                half_reaction_0["create_bonds"], half_reaction_1["create_bonds"]
            )
            # in theory, this is simple
            for ix_0, ix_1 in bonds_to_make:
                atom_0, atom_1 = topology_atoms[ix_0], topology_atoms[ix_1]
                topology.addBond(atom_0, atom_1)
        return topology

    @staticmethod
    def _delete_bonds_and_atoms(topology, topology_atoms, reactions, positions):
        """
        Creates a Modeller object and uses it to delete atoms and bonds in the half reactions.

        Args:
            topology: an OpenMM topology
            topology_atoms: a dictionary of atoms in the topology, keys are ids, values are Atoms
            reactions: tuples of half_reaction dicts
            positions: the physical positions of each atom, needed to instantiate a Modeler

        Returns:

        """
        modeller = Modeller(topology, positions)
        bonds_to_delete = []
        atoms_to_delete = []
        for half_reaction_0, half_reaction_1 in reactions:
            # generate bonds to delete
            bonds_to_delete_ix = (
                half_reaction_0["delete_bonds"] + half_reaction_1["delete_bonds"]
            )
            for ix_0, ix_1 in bonds_to_delete_ix:
                atom_0, atom_1 = topology_atoms[ix_0], topology_atoms[ix_1]
                bonds_to_delete.append((atom_0, atom_1))
            # generate atoms to delete
            atoms_to_delete_ix = (
                half_reaction_0["delete_atoms"] + half_reaction_1["delete_atoms"]
            )
            for ix_0, ix_1 in atoms_to_delete_ix:
                atom_0, atom_1 = topology_atoms[ix_0], topology_atoms[ix_1]
                atoms_to_delete.append((atom_0, atom_1))
        # delete bonds and atoms
        modeller.delete(bonds_to_delete + atoms_to_delete)
        return modeller.getTopology()

    def _update_topology(self, topology, positions, cutoff_distance):
        """
        A single cycle of the reaction scheme.

        Args:
            topology: an OpenMM topology
            positions: the positions of each atom in the topology.
            cutoff_distance: cutoff distance to be used on the trigger atoms
        """
        topology_atoms = {
            self.index_map[i]: atom for i, atom in enumerate(topology.atoms())
        }
        atoms_ix_0 = self.index_map[self.trigger_atoms[0]]
        atoms_ix_1 = self.index_map[self.trigger_atoms[1]]
        # TODO: will return indices in atoms_0 array, not in topology. Need to reindex in loop.
        positions_0, positions_1 = positions[atoms_ix_0], positions[atoms_ix_1]
        reaction_pairs = capped_distance(
            positions_0, positions_1, cutoff_distance, return_distances=False
        )
        reactions = []
        for ix_0, ix_1 in reaction_pairs:
            half_reaction_0 = self.half_reactions[atoms_ix_0[ix_0]]
            half_reaction_1 = self.half_reactions[atoms_ix_1[ix_1]]
            reactions.append((half_reaction_0, half_reaction_1))
        # TODO: we never remove the atoms from self.trigger_atoms!!!!!!
        topology = self._create_bonds(topology, topology_atoms, reactions)
        topology = self._delete_bonds_and_atoms(
            topology, topology_atoms, reactions, positions
        )
        return topology

    def _update_input_files(self):
        """
        Internal method to update the topology and system files

        Returns:

        """
