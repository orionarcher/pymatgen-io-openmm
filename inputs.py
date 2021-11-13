import io
import os
import re
import shutil
import warnings
from pathlib import Path
from string import Template
from typing import Union, Optional, Dict, List

import numpy as np
from monty.json import MSONable
from monty.dev import deprecated

import pymatgen.core
from pymatgen.io.core import InputFile, InputSet, InputGenerator

from openmm.app import Simulation, PDBFile, Topology
from openmm import XmlSerializer, System, Integrator, State

from pymatgen.io.babel import BabelMolAdaptor
from openbabel import pybel


__author__ = "Orion Cohen"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


class TopologyInput(InputFile):
    def __init__(self, topology: Topology, positions: Union[np.ndarray, List]):
        self.topology = topology
        self.positions = positions

    def get_string(self) -> str:
        with io.StringIO() as s:
            PDBFile.writeFile(self.topology, self.positions)
            s.seek(0)
            pdb = s.read()
        return pdb

    @classmethod
    def from_string(cls, contents: str):
        with io.StringIO(contents) as s:
            pdb = PDBFile(s)
            topology = pdb.getTopology()
            positions = pdb.getPositions(asNumpy=True)
        return TopologyInput(topology, positions)


class SystemInput(InputFile):
    def __init__(self, system: System):
        self.system = system

    def get_string(self) -> str:
        return XmlSerializer.serialize(self.system)

    @classmethod
    def from_string(cls, contents: str):
        return SystemInput(XmlSerializer.deserialize(contents))


class IntegratorInput(InputFile):
    def __init__(self, integrator: Integrator):
        self.integrator = integrator

    def get_string(self) -> str:
        return XmlSerializer.serialize(self.integrator)

    @classmethod
    def from_string(cls, contents: str):
        return IntegratorInput(XmlSerializer.deserialize(contents))


class StateInput(InputFile):
    def __init__(self, state: State):
        self.state = state

    def get_string(self) -> str:
        return XmlSerializer.serialize(self.state)

    @classmethod
    def from_string(cls, contents: str):
        return StateInput(XmlSerializer.deserialize(contents))


class OpenMMSet(InputSet):

    # TODO: if there is an optional file should it be missing or should the value be none?
    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        topology_file: str = "topology.pdb",
        system_file: str = "system.xml",
        integrator_file: str = "integrator.xml",
        state_file: str = "state.xml",
    ):
        topology = TopologyInput.from_file(topology_file)
        system = TopologyInput.from_file(system_file)
        integrator = TopologyInput.from_file(integrator_file)
        openmm_set = OpenMMSet(
            topology=topology,
            system=system,
            integrator=integrator,
        )
        if Path("state.xml").is_file():
            state = TopologyInput.from_file(state_file)
            openmm_set["state"] = state
        return openmm_set

    def validate(self) -> bool:
        # TODO: this should test if the set returns a valid simulation and throw an error if it does not
        # TODO: is this condition too strict?
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

    def get_simulation(self) -> Simulation:
        simulation = Simulation(
            self.topology.topology,
            self.system.system,
            self.integrator.integrator,
        )
        if hasattr(self, "state") and self.state:
            # TODO: confirm that this works correctly
            simulation.context.setState(self.state.state)
        return simulation


# noinspection PyMethodOverriding
class OpenMMGenerator(InputGenerator):

    # TODO: what determines if a setting goes in the __init__ or get_input_set?
    def __init__(
        self,
        force_field: str = "Sage",
        integrator: Union[str, Integrator] = "LangevinMiddleIntegrator",
        temperature: float = 298,
        step_size: int = 1,
        topology_file: Union[str, Path] = "topology.pdb",
        system_file: Union[str, Path] = "system.xml",
        integrator_file: Union[str, Path] = "integrator.xml",
        state_file: Union[str, Path] = "state.xml",
    ):
        return

    def get_input_set(
        self,
        molecules: Dict[Union[str], int],
        density: Optional[float] = None,
        box: Optional[List] = None,
        temperature: Optional[float] = None,
    ) -> InputSet:


        return


    def _smile_to_molecule(self, smile):
        """
        Converts a SMILE to a Pymatgen Molecule.
        """
        mol = pybel.readstring("smi", smile)
        mol.addh()
        mol.make3D()
        adaptor = BabelMolAdaptor(mol.OBMol)
        return adaptor.pymatgen_mol
