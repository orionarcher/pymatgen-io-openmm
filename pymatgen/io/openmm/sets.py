"""
Concrete implementations of InputSet for the OpenMM IO.
"""

# base python
from pathlib import Path
from typing import Union, Optional, Dict

from monty.serialization import loadfn

# openmm
import openmm
from openmm.app import Simulation

import MDAnalysis as mda

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
        System XML, a Integrator XML, and (optionally) a State XML. If no State
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
        source_dir = Path(directory)
        topology_input = TopologyInput.from_file(source_dir / topology_file)
        system_input = SystemInput.from_file(source_dir / system_file)
        integrator_input = IntegratorInput.from_file(source_dir / integrator_file)
        inputs = {
            topology_file: topology_input,
            system_file: system_input,
            integrator_file: integrator_input,
        }
        openmm_set = OpenMMSet(
            inputs=inputs,  # type: ignore
            topology_file=topology_file,
            system_file=system_file,
            integrator_file=integrator_file,
        )
        if Path(source_dir / state_file).is_file():
            openmm_set.inputs[state_file] = StateInput.from_file(
                source_dir / state_file
            )
            openmm_set.state_file = state_file  # should this be a dict-like assignment?
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
        platform: Optional[Union[str, openmm.Platform]] = None,
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
            platform = openmm.Platform.getPlatformByName(platform)
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
    def from_directory(
        cls, directory: Union[str, Path], rxn_atoms_file="rxn_atoms.json", force_field_file="force_field.json", **kwargs
    ):
        input_set = super().from_directory(directory, **kwargs)
        input_set.inputs = {
            **input_set.inputs,
            rxn_atoms_file: loadfn(rxn_atoms_file),
            force_field_file: loadfn(force_field_file),
        }
        input_set.rxn_atoms_file = rxn_atoms_file
        input_set.force_field_file = force_field_file
        input_set.__class__ = OpenMMAlchemySet
        return input_set

    def run(self, n_cycles):
        """
        Run the networking procedure for n_cycles.

        Args:
            cycles:

        Returns:

        """
        for i in range(len(n_cycles)):
            self.single_cycle()
        return 1

    def get_universe(self):
        """
        Returns an mda universe of the simulation. Useful for debugging atom selections.

        Returns:

        """
        topology = self.inputs[self.topology_file]
        topology = topology.get_topology()
        u = mda.Universe(topology)
        return u

    def get_atom_group(self, group_name):
        """
        Creates an MDAnalysis Universe and queries it to get the dynamic atoms

        Returns:

        """
        u = self.get_universe()
        group = u.select_atoms("string")
        return group

    def single_cycle(self):
        """
        A single cycle of the reaction scheme.
        """

    def _update_input_files(self):
        """
        Internal method to update the topology and system files

        Returns:

        """
