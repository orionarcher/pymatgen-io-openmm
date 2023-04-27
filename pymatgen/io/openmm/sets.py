"""
Concrete implementations of InputSet for the OpenMM IO.
"""

# base python
from pathlib import Path
from typing import Union, Optional, Dict


# openmm
import openmm
from openmm.app import Simulation


# pymatgen
from pymatgen.io.core import InputSet
from pymatgen.io.openmm.inputs import (
    TopologyInput,
    SystemInput,
    IntegratorInput,
    StateInput,
    ReactiveSystemInput,
    SetContentsInput,
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
        contents_file: str = "contents.json",
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
            contents_file: name of the json file that holds the molecules in the set.


        Returns:
            an OpenMMSet
        """
        # TODO: will need to add some sort of settings file
        source_dir = Path(directory)
        topology_input = TopologyInput.from_file(source_dir / topology_file)
        system_input = SystemInput.from_file(source_dir / system_file)
        integrator_input = IntegratorInput.from_file(source_dir / integrator_file)
        state_input = StateInput.from_file(source_dir / state_file)
        contents_input = SetContentsInput.from_file(source_dir / contents_file)
        inputs = {
            topology_file: topology_input,
            system_file: system_input,
            integrator_file: integrator_input,
            state_file: state_input,
            contents_file: contents_input,
        }
        openmm_set = OpenMMSet(
            inputs=inputs,  # type: ignore
            topology_file=topology_file,
            system_file=system_file,
            integrator_file=integrator_file,
            state_file=state_file,
            contents_file=contents_file,
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
        state_input = self.inputs[self.state_file]
        if isinstance(platform, str):
            platform = openmm.openmm.Platform.getPlatformByName(platform)  # type: ignore
        simulation = Simulation(
            topology_input.get_topology(),  # type: ignore
            system_input.get_system(),  # type: ignore
            integrator_input.get_integrator(),  # type: ignore
            platform=platform,
            platformProperties=platformProperties,
        )
        simulation.context.setState(state_input.get_state())
        return simulation


class OpenMMAlchemySet(OpenMMSet):
    """
    An InputSet for an alchemical reaction workflow.
    """

    @classmethod
    def from_directory(cls, directory: Union[str, Path], reactive_system_file="reactive_system.json", **kwargs):  # type: ignore
        input_set = super().from_directory(directory, **kwargs)
        source_dir = Path(directory)
        reactive_system_input = ReactiveSystemInput.from_file(
            source_dir / reactive_system_file
        )

        # the json will store the int keys as strings, so we need to convert them back
        for (
            reactive_atom_set
        ) in reactive_system_input.reactive_system.reactive_atom_sets:
            reactive_atom_set.half_reactions = {
                int(k): v for k, v in reactive_atom_set.half_reactions.items()
            }

        input_set.inputs = {
            **input_set.inputs,
            reactive_system_file: reactive_system_input,
        }
        input_set.reactive_system_file = reactive_system_file
        input_set.__class__ = OpenMMAlchemySet
        return input_set
