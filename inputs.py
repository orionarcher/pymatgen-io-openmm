"""
Concrete implementations of InputFiles for the OpenMM IO.
"""

# base python
import io
from typing import Union, Optional, List

# cheminformatics
import numpy as np

# openmm
from openmm.app import PDBFile, Topology
from openmm import (
    XmlSerializer,
    System,
    Integrator,
    State,
)

# pymatgen
from pymatgen.io.core import InputFile

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


class TopologyInput(InputFile):
    """
    Input handler for OpenMM topologies. Stores and parses PDB files.
    """

    def __init__(self, topology: Topology, positions: Optional[Union[List, np.ndarray]] = None):
        """
        Instatiates a TopologyInput from an OpenMM.Topology. Positions can be supplied as
        a n x 3 numpy array. If they are not given, positions will be set to 0.

        Args:
            topology: the openmm topology to serialize
            positions: coordinates for each particle in the topology
        """
        self.content = self._serialize(topology, positions)

    @staticmethod
    def _serialize(topology, positions) -> str:
        if not positions:
            positions = np.zeros(shape=(topology.getNumAtoms(), 3))
        with io.StringIO() as s:
            PDBFile.writeFile(topology, positions, file=s)
            s.seek(0)
            pdb = s.read()
        return pdb

    def get_string(self) -> str:
        """
        Get a string representation of the topology PDB.

        Returns:
            A string representation of the PDB topology file.
        """
        return self.content

    @classmethod
    def from_string(cls, contents: str) -> InputFile:
        """
        Get an InputFile from a PDB string.

        Args:
            contents: the contents of a PDB file.

        Returns:
            An InputFile representing the topology pdb.
        """
        with io.StringIO(contents) as s:
            pdb = PDBFile(s)
            topology = pdb.getTopology()
        return TopologyInput(topology)

    def get_topology(self) -> Topology:
        """
        Returns the OpenMM topology represented by the TopologyInput.

        Returns:
            openmm.app.Topology
        """
        with io.StringIO(self.content) as s:
            pdb = PDBFile(s)
            topology = pdb.getTopology()
        return topology


class XmlInput(InputFile):
    """
    A standardized definition for InputFiles based on OpenMM XML serialization.

    compatible with any OpenMM object with a serialization proxy registered:
    https://github.com/openmm/openmm/blob/master/serialization/src/SerializationProxyRegistration.cpp
    """

    def __init__(self, openmm_object):
        """
        Create an InputFile from a serializable OpenMM object.

        Args:
            openmm_object:
        """
        self.content = self._serialize(openmm_object)

    @staticmethod
    def _serialize(openmm_object) -> str:
        return XmlSerializer.serialize(openmm_object)

    def get_string(self) -> str:
        """
        Return a string of the serialized Xml file.

        Returns:
            string
        """
        return self.content

    @classmethod
    def from_string(cls, contents: str):
        """
        This is a template that should be overwritten. Replace XmlInput with child class.
        """
        return XmlInput(XmlSerializer.deserialize(contents))


class SystemInput(XmlInput):
    """
    Input handler for OpenMM systems. Stores and parses XML files.
    """

    @classmethod
    def from_string(cls, contents: str) -> InputFile:
        """
        Return a SystemInput from a serialized XML System file.

        Args:
            contents: the XML System string

        Returns:
            SystemInput object
        """
        return SystemInput(XmlSerializer.deserialize(contents))

    def get_system(self) -> System:
        """
        Returns the OpenMM system represented by the SystemInput.

        Returns:
            openmm.System
        """
        return XmlSerializer.deserialize(self.content)


class IntegratorInput(XmlInput):
    """
    Input handler for OpenMM integrators. Stores and parses XML files.
    """

    @classmethod
    def from_string(cls, contents: str) -> InputFile:
        """
        Return a IntegratorInput from a serialized XML Integrator file.

        Args:
            contents: the XML Integrator string

        Returns:
            IntegratorInput object
        """
        return IntegratorInput(XmlSerializer.deserialize(contents))

    def get_integrator(self) -> Integrator:
        """
        Returns the OpenMM integrator represented by the IntegratorInput.

        Returns:
            openmm.Integrator
        """
        return XmlSerializer.deserialize(self.content)


class StateInput(XmlInput):
    """
    State handler for OpenMM integrators. Stores and parses XML files.
    """

    @classmethod
    def from_string(cls, contents: str) -> InputFile:
        """
        Return a StateInput from a serialized XML State file.

        Args:
            contents: the XML State string

        Returns:
            StateInput object
        """
        return StateInput(XmlSerializer.deserialize(contents))

    def get_state(self) -> State:
        """
        Returns the OpenMM state represented by the StateInput.

        Returns:
            openmm.State
        """
        return XmlSerializer.deserialize(self.content)
