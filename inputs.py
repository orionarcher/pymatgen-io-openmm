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



class TopologyInputFile(InputFile):
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
        return pdb

class SystemInputFile(InputFile):
    def __init__(self, system: System):
        self.system = system

    def get_string(self) -> str:
        return XmlSerializer.serialize(self.system)

    @classmethod
    def from_string(cls, contents: str):
        return SystemInputFile(XmlSerializer.deserialize(contents))


class IntegratorInputFile(InputFile):
    def __init__(self, integrator: Integrator):
        self.integrator = integrator

    def get_string(self) -> str:
        return XmlSerializer.serialize(self.integrator)

    @classmethod
    def from_string(cls, contents: str):
        return SystemInputFile(XmlSerializer.deserialize(contents))


class StateInputFile(InputFile):
    def __init__(self, state: State):
        self.state = state

    def get_string(self) -> str:
        return XmlSerializer.serialize(self.state)

    @classmethod
    def from_string(cls, contents: str):
        return SystemInputFile(XmlSerializer.deserialize(contents))



class OpenMMSet(InputSet):

    def from_directory(cls, directory: Union[str, Path]):
        return

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
