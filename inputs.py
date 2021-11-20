import io
import os
from pathlib import Path
import pathlib
import re
import shutil
import tempfile
import warnings
from string import Template
from typing import Union, Optional, Dict, List

import numpy as np
from monty.json import MSONable
from monty.dev import deprecated

import pymatgen.core
from pymatgen.io.core import InputFile, InputSet, InputGenerator
from pymatgen.io.packmol import PackmolBoxGen
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.xyz import XYZ

import openff
import openff.toolkit
from openff.toolkit.typing.engines import smirnoff

import openmm
from openmm.app import Simulation, PDBFile, Topology
from openmm import (
    XmlSerializer,
    System,
    Integrator,
    State,
    Context,
    LangevinMiddleIntegrator,
)
from openmm.unit import *

from openbabel import pybel

import parmed

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


class TopologyInput(InputFile):
    def __init__(self, topology: Topology, positions: Optional[Union[List, np.ndarray]] = None):
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
        return self.content

    @classmethod
    def from_string(cls, contents: str):
        with io.StringIO(contents) as s:
            pdb = PDBFile(s)
            topology = pdb.getTopology()
        return TopologyInput(topology)

    def get_topology(self) -> Topology:
        with io.StringIO(self.content) as s:
            pdb = PDBFile(s)
            topology = pdb.getTopology()
        return topology

class XmlInput(InputFile):
    """
    compatible with any OpenMM object with a serialization proxy registered:
    https://github.com/openmm/openmm/blob/master/serialization/src/SerializationProxyRegistration.cpp
    """
    def __init__(self, openmm_object):
        self.content = self._serialize(openmm_object)

    @staticmethod
    def _serialize(openmm_object) -> str:
        return XmlSerializer.serialize(openmm_object)

    def get_string(self) -> str:
        return self.content

    @classmethod
    def from_string(cls, contents: str):
        return XmlInput(XmlSerializer.deserialize(contents))


class SystemInput(XmlInput):
    @classmethod
    def from_string(cls, contents: str):
        return SystemInput(XmlSerializer.deserialize(contents))

    def get_system(self) -> System:
        return XmlSerializer.deserialize(self.content)


class IntegratorInput(XmlInput):
    @classmethod
    def from_string(cls, contents: str):
        return IntegratorInput(XmlSerializer.deserialize(contents))

    def get_integrator(self) -> Integrator:
        return XmlSerializer.deserialize(self.content)


class StateInput(XmlInput):
    @classmethod
    def from_string(cls, contents: str):
        return StateInput(XmlSerializer.deserialize(contents))

    def get_state(self) -> State:
        return XmlSerializer.deserialize(self.content)


class OpenMMSet(InputSet):
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

        Args:
            directory:
            topology_file: name of the pdb file with topological information.
            system_file: name of the serialized System xml file.
            integrator_file: name of the serialized Integrator xml file.
            state_file: name of the serialized State xml file. If there is no state_file,
                then positions must be set for simulation.

        Returns:
            an OpenMMSet
        """
        dir = Path(directory)
        topology_input = TopologyInput.from_file(dir / topology_file)
        system_input = SystemInput.from_file(dir / system_file)
        integrator_input = IntegratorInput.from_file(dir / integrator_file)
        inputs = {
            topology_file: topology_input,
            system_file: system_input,
            integrator_file: integrator_input,
        }
        openmm_set = OpenMMSet(
            inputs=inputs,
            topology_file=topology_file,
            system_file=system_file,
            integrator_file=integrator_file,
        )
        if Path(dir / state_file).is_file():
            openmm_set.inputs[state_file] = StateInput.from_file(dir / state_file)
            openmm_set.state_file = state_file  # should this be a dict-like assignment?
        return openmm_set

    def validate(self) -> bool:
        """
        Check that the OpenMMSet will generate a valid simulation.

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

    def get_simulation(self) -> Simulation:
        """
        Instantiates and returns an OpenMM.Simulation from the input files.

        Returns:
            OpenMM.Simulation
        """
        topology_input = self.inputs[self.topology_file]
        system_input = self.inputs[self.system_file]
        integrator_input = self.inputs[self.integrator_file]
        simulation = Simulation(
            topology_input.get_topology(),
            system_input.get_system(),
            integrator_input.get_integrator(),
        )
        if hasattr(self, "state_file") and self.state_file:
            # TODO: confirm that this works correctly
            state_input = self.inputs[self.state_file]
            simulation.context.setState(state_input.get_state())
        return simulation


# noinspection PyMethodOverriding
class OpenMMGenerator(InputGenerator):
    """
    Generator for an OpenMM InputSet that specifies a simulation of a mixed molecular system.

    This class is only compatible with the Langevin Middle Integrator. To use a different
    integrator, you can first generate the system with OpenMMGenerator and then add
    a different integrator to the OpenMMInputSet.
    """

    def __init__(
        self,
        force_field: str = "Sage",
        temperature: float = 298,
        step_size: float = 0.001,
        friction_coefficient: int = 1,
        partial_charges: Optional[Dict[str, np.ndarray]] = None,
        topology_file: Union[str, Path] = "topology.pdb",
        system_file: Union[str, Path] = "system.xml",
        integrator_file: Union[str, Path] = "integrator.xml",
        state_file: Union[str, Path] = "state.xml",
    ):
        """
        Instantiates an OpenMMGenerator.

        Args:
            force_field: force field for parameterization, currently supported Force Fields: 'Sage'.
            temperature: the temperature to be added to the integrator (Kelvin).
            step_size: the step size of the simulation (picoseconds).
            friction_coefficient: the friction coefficient which couples the system to
                the heat bath (inverse picoseconds).
            partial_charges: TODO: determine a appropriate way of specifying partial charge information
            topology_file: Location to save the Topology PDB.
            system_file: Location to save the System xml.
            integrator_file: Location to save the Integrator xml.
            state_file: Location to save the State xml.
        """
        self.force_field = force_field
        self.temperature = temperature
        self.step_size = step_size
        self.friction_coefficient = friction_coefficient
        self.partial_charges = partial_charges
        self.topology_file = topology_file
        self.system_file = system_file
        self.integrator_file = integrator_file
        self.state_file = state_file

    def get_input_set(
        self,
        smiles: Dict[str, int],
        density: Optional[float] = None,
        box: Optional[List] = None,
        temperature: Optional[float] = None,
    ) -> InputSet:
        """

        Args:
            smiles: keys are smiles and values are number of that molecule to pack
            density: the density of the system. density OR box must be given as an argument.
            box: list of [xlo, ylo, zlo, xhi, yhi, zhi]. density OR box must be given as an argument.
            temperature: the temperature of the system (Kelvin).

        Returns:
            an OpenMM.InputSet
        """
        assert (density is None) ^ (box is None), "Density OR box must be included, but not both."
        # TODO: write test to ensure coordinates and topology have the same atom ordering
        # create dynamic openmm objects with internal methods
        topology = self._get_openmm_topology(smiles)
        if not box:
            box = self._get_box(smiles, density)
        coordinates = self._get_coordinates(smiles, box)
        smile_strings = list(smiles.keys())
        system = self._parameterize_system(
            topology, smile_strings, box, self.force_field
        )
        integrator = LangevinMiddleIntegrator(
            self.temperature * kelvin,
            self.friction_coefficient / picoseconds,
            self.step_size * picoseconds,
        )
        context = Context(system, integrator)
        context.setPositions(coordinates)
        state = context.getState(getPositions=True)
        # instantiate input files and feed to input_set
        topology_input = TopologyInput(topology)
        system_input = SystemInput(system)
        integrator_input = IntegratorInput(integrator)
        state_input = StateInput(state)
        input_set = OpenMMSet(
            inputs={
                self.topology_file: topology_input,
                self.system_file: system_input,
                self.integrator_file: integrator_input,
                self.state_file: state_input,
            },
            topology_file=self.topology_file,
            system_file=self.system_file,
            integrator_file=self.integrator_file,
            state_file=self.state_file,
        )
        return input_set

    @staticmethod
    def _smile_to_molecule(smile: str) -> pymatgen.core.Molecule:
        """
        Convert a SMILE to a Pymatgen.Molecule.
        """
        mol = pybel.readstring("smi", smile)
        mol.addh()
        mol.make3D()
        adaptor = BabelMolAdaptor(mol.OBMol)
        return adaptor.pymatgen_mol

    @staticmethod
    def _smile_to_parmed_structure(smile: str) -> parmed.Structure:
        """
        Convert a SMILE to a Parmed.Structure.
        """
        mol = pybel.readstring("smi", smile)
        mol.addh()
        mol.make3D()
        with tempfile.NamedTemporaryFile() as f:
            mol.write(format="mol", filename=f.name, overwrite=True)
            structure = parmed.load_file(f.name)[0]  # load_file is returning a list for some reason
        return structure

    @staticmethod
    def _get_openmm_topology(smiles: Dict[str, int]) -> openmm.app.Topology:
        """
        Returns an openmm topology with the given smiles at the given counts.

        The topology does not contain coordinates.

        Parameters:
            smiles: keys are smiles and values are number of that molecule to pack

        Returns:
            an openmm.app.Topology
        """
        structure_counts = {
            count: OpenMMGenerator._smile_to_parmed_structure(smile)
            for smile, count in smiles.items()
        }
        combined_structs = parmed.Structure()
        for struct, count in structure_counts.items():
            combined_structs += struct * count
        return combined_structs.topology

    @staticmethod
    def _get_coordinates(smiles: Dict[str, int], box: List[float]) -> np.ndarray:
        """
        Pack the box with the molecules specified by smiles.

        Args:
            smiles: keys are smiles and values are number of that molecule to pack
            box: list of [xlo, ylo, zlo, xhi, yhi, zhi]

        Returns:
            array of coordinates for each atom in the box.
        """
        molecules = []
        for smile, count in smiles.items():
            molecules.append(
                {
                    "name": smile,
                    "number": count,
                    "coords": OpenMMGenerator._smile_to_molecule(smile),
                }
            )
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolBoxGen().get_input_set(molecules=molecules, box=box)
            pw.write_input(scratch_dir)
            pw.run(scratch_dir)
            coordinates = XYZ.from_file(
                pathlib.Path(scratch_dir, "packmol_out.xyz")
            ).as_dataframe()
        raw_coordinates = coordinates.loc[:, "x":"z"].values
        return raw_coordinates

    @staticmethod
    def _get_box(smiles: Dict[str, int], density: float) -> List[float]:
        """
        Calculates the side_length of a cube necessary to contain the given molecules with
        given density.

        Args:
            smiles: keys are smiles and values are number of that molecule to pack
            density: guessed density of the solution, larger densities will lead to smaller cubes.

        Returns:
            side_length: side length of the returned cube
        """
        cm3_to_A3 = 1e24
        NA = 6.02214e23
        mols = [OpenMMGenerator._smile_to_molecule(smile) for smile in smiles.keys()]
        mol_mw = np.array([mol.composition.weight for mol in mols])
        counts = np.array(list(smiles.values()))
        total_weight = sum(mol_mw * counts)
        box_volume = total_weight * cm3_to_A3 / (NA * density)
        side_length = round(box_volume ** (1 / 3), 2)
        return [0, 0, 0, side_length, side_length, side_length]

    # TODO: need to settle on a method for selecting a parameterization of the system.
    # TODO: this code should be restructured to take topology, smiles, box, and ff as args
    @staticmethod
    def _parameterize_system(
        topology: Topology, smile_strings: List[str], box: List[float], force_field: str
    ) -> openmm.System:
        supported_force_fields = ["Sage"]
        if force_field.lower() == "sage":
            openff_mols = [
                openff.toolkit.topology.Molecule.from_smiles(smile)
                for smile in smile_strings
            ]
            # TODO: add logic to insert partial charges into ff
            openff_forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
            openff_topology = openff.toolkit.topology.Topology.from_openmm(
                topology, openff_mols
            )
            box_vectors = list(np.array(box[3:6]) - np.array(box[0:3])) * angstrom
            openff_topology.box_vectors = box_vectors
            system = openff_forcefield.create_openmm_system(openff_topology)
            return system
        else:
            raise NotImplementedError(
                f"currently only these force fields are supported: {' '.join(supported_force_fields)}.\n"
                f"Please select one of the supported force fields."
            )
