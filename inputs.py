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
from openff.toolkit.typing.engines import smirnoff

import openmm
from openmm.app import Simulation, PDBFile, Topology
from openmm import XmlSerializer, System, Integrator, State
from openmm.unit import *

from openbabel import pybel

import parmed

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
        partial_charges: Optional[Dict[str, np.ndarray]] = None,
        topology_file: Union[str, Path] = "topology.pdb",
        system_file: Union[str, Path] = "system.xml",
        integrator_file: Union[str, Path] = "integrator.xml",
        state_file: Union[str, Path] = "state.xml",
    ):
        return

    def get_input_set(
        self,
        smiles: Dict[str, int],
        density: Optional[float] = None,
        box: Optional[List] = None,
        temperature: Optional[float] = None,
    ) -> InputSet:
        # the way these functions are written write now is not a pipeline, each internal
        # method should be called to generate the next step in the pipe, not all take
        # the same methods. e.g. the static utility methods should not call eachother
        # unless strictly necessary, instead, get_input_set should string together the
        # operations to create a clean pipeline.
        return

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
            mol.write(format="pdb", filename=f.name, overwrite=True)
            structure = parmed.load_file(f.name)
        return structure

    @staticmethod
    def _get_openmm_topology(smiles: Dict[str, int]) -> openmm.app.Topology:
        """
        Returns an openmm topology with the given smiles at the given counts.

        The topology does not contain coordinates.

        Parameters
        ----------
        smiles : keys are smiles and values are number of that molecule to pack

        Returns
        -------
        topology
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

        Parameters
        ----------
        smiles : keys are smiles and values are number of that molecule to pack
        density : guessed density of the solution, larger densities will lead to smaller cubes.

        Returns
        -------
        side_length: side length of the returned cube
        """
        cm3_to_A3 = 1e24
        NA = 6.02214e23
        mols = [OpenMMGenerator._smile_to_molecule(smile) for smile in smiles.keys()]
        mol_mw = np.array([mol.composition.weight for mol in mols])
        counts = np.array(list(smiles.values()))
        total_weight = sum(mol_mw * counts)
        box_volume = total_weight * cm3_to_A3 / (NA * density)
        side_length = box_volume ** (1 / 3)
        return [0, 0, 0, side_length, side_length, side_length]

    # TODO: need to settle on a method for selecting a parameterization of the system.
    # TODO: this code should be restructured to take topology, smiles, box, and ff as args
    @staticmethod
    def _parameterize_system(
        smiles: Dict[str, int], box: List[float], force_field: str
    ) -> openmm.System:
        supported_force_fields = ["Sage"]
        if force_field == "Sage":
            topology = OpenMMGenerator._get_openmm_topology(smiles)
            openff_mols = [
                openff.toolkit.topology.Molecule.from_smiles(smile)
                for smile in smiles.keys()
            ]
            # TODO: add logic to insert partial charges into ff
            openff_forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
            openff_topology = openff.toolkit.topology.Topology.from_openmm(
                topology, openff_mols
            )
            box_vectors = list(np.array(box[3:6]) - np.array(box[0:3])) * nanometer
            openff_topology.box_vectors = box_vectors
            system = openff_forcefield.create_openmm_system(openff_topology)
            return system
        else:
            raise NotImplementedError(
                f"currently only these force fields are supported: {' '.join(supported_force_fields)}.\n"
                f"Please select one of the supported force fields."
            )
