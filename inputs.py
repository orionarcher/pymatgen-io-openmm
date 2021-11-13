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

from pymatgen.io.core import InputFile, InputSet, InputGenerator

from openmm.app import Simulation, PDBFile, Topology
from openmm import XmlSerializer, System, Integrator, State


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
    def from_directory(cls,
                       directory: Union[str, Path],
                       topology_file: str = "topology.pdb",
                       system_file: str = "system.xml",
                       integrator_file: str = "integrator.xml",
                       state_file:str = "state.xml",
                       ):
        topology = TopologyInput.from_file(topology_file)
        system = TopologyInput.from_file(system_file)
        integrator = TopologyInput.from_file(integrator_file)
        openmm_set = OpenMMSet(
            topology=topology,
            system=system,
            integrator=integrator,
        )
        if Path('state.xml').is_file():
            state = TopologyInput.from_file(state_file)
            openmm_set['state'] = state
        return openmm_set

    def validate(self) -> bool:
        return False

    def get_simulation(self) -> Simulation:
        return



class OpenMMGenerator(InputGenerator):

    # TODO: what determines if a setting goes in the __init__ or get_input_set?
    def __init__(self,
                 force_field,
                 integrator,
                 temperature,
                 step_size,):

    def get_input_set(self) -> InputSet:
        return
