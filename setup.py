from abc import ABC

import openmm.app
from openbabel import pybel
from openff.toolkit.typing.engines import smirnoff
from openmm.app import Simulation
from pymatgen.io.babel import BabelMolAdaptor

from openmm.app.topology import Topology
from openmm import LangevinMiddleIntegrator

from pymatgen.io.packmol import PackmolBoxGen
from pymatgen.io.xyz import XYZ

from pymatgen.io.core import InputGenerator

from openmm.unit import *

import numpy as np
import parmed
import openff

import pathlib
import tempfile
from typing import Dict, Union, Optional


# does this belong as an attribute of the Molecule class?
def smile_to_mol(smile):
    """
    Converts a SMILE to a Pymatgen Molecule.

    Parameters
    ----------
    smile: a SMILE.

    Returns
    -------
    Pymatgen.Molecule
    """
    mol = pybel.readstring("smi", smile)
    mol.addh()
    mol.make3D()
    adaptor = BabelMolAdaptor(mol.OBMol)
    return adaptor.pymatgen_mol


class OpenMMSimulationGenerator:
    """
    An opinionated generator for OpenMM Simulations sets.

    All parameters are serializable and OpenMM objects will only be instantiated
    when the return_simulation function is called.
    """

    def __init__(
        self,
        smile_counts: Dict[str, int],
        density: float,
        integrator: Optional[str] = None,
        platform: str = None,
        platformProperties: Optional[Dict] = None,
        state: Optional[Union[str, pathlib.Path]] = None,
    ):
        self.smile_counts = smile_counts
        self.density = density
        self.integrator = LangevinMiddleIntegrator if not integrator else integrator
        self.platform = platform
        self.platformProperties = platformProperties
        self.state = state

    def _smile_to_parmed_structure(self, smile: str) -> parmed.Structure:
        """
        Converts a SMILE to a Parmed structure.

        Parameters
        ----------
        smile: a SMILE.

        Returns
        -------
        Parmed.Structure
        """
        mol = pybel.readstring("smi", smile)
        mol.addh()
        mol.make3D()
        with tempfile.NamedTemporaryFile() as f:
            mol.write(format="pdb", filename=f.name, overwrite=True)
            structure = parmed.load_file(f.name)
        return structure

    def _smiles_to_openmm_topology(
        self, smile_counts: Dict[str, int]
    ) -> openmm.app.Topology:
        """
        Returns an openmm topology with the given smiles at the given counts.

        The topology does not contain coordinates.

        Parameters
        ----------
        smile_counts : keys are smiles and values are number of that molecule to pack

        Returns
        -------
        topology
        """
        structure_counts = {
            count: self._smile_to_parmed_structure(smile)
            for smile, count in smile_counts.values()
        }
        combined_structs = parmed.Structure()
        for struct, count in structure_counts:
            combined_structs += struct * count
        return combined_structs.topology

    def _smiles_to_coordinates(
        self, smile_counts: Dict[str, int], box_size: float
    ) -> np.ndarray:
        molecules = []
        for smile, count in smile_counts.items():
            molecules.append(
                {"name": smile, "number": count, "coords": smile_to_mol(smile)}
            )
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolBoxGen().get_input_set(
                molecules=molecules, box=[0, 0, 0, box_size, box_size, box_size]
            )
            pw.write_input(scratch_dir)
            pw.run(scratch_dir)
            coordinates = XYZ.from_file(
                pathlib.Path(scratch_dir, "packmol_out.xyz")
            ).as_dataframe()
        raw_coordinates = coordinates.loc[:, "x":"z"].values
        return raw_coordinates

    def _smiles_to_cube_size(
        self, smile_counts: Dict[str, int], density: float
    ) -> float:
        """
        Calculates the side_length of a cube necessary to contain the given molecules with
        given density.

        Parameters
        ----------
        smile_counts : keys are smiles and values are number of that molecule to pack
        density : guessed density of the solution, larger densities will lead to smaller cubes.

        Returns
        -------
        side_length: side length of the returned cube
        """
        cm3_to_A3 = 1e24
        NA = 6.02214e23
        mols = [smile_to_mol(smile) for smile in smile_counts.keys()]
        mol_mw = np.array([mol.structure.composition.weight for mol in mols])
        counts = np.array(smile_counts.values())
        total_weight = sum(mol_mw * counts)
        box_volume = total_weight * cm3_to_A3 / (NA * density)
        side_length = box_volume ** (1 / 3)
        return side_length

    def _smiles_to_system_topology(
        self, smile_counts: Dict[str, int], density: float = 1.5
    ) -> openmm.System:
        """

        Parameters
        ----------
        smile_counts
        density

        Returns
        -------

        """
        topology = self._smiles_to_openmm_topology(smile_counts)
        box_size = self._smiles_to_cube_size(smile_counts, density)
        openff_mols = [
            openff.toolkit.topology.Molecule.from_smiles(smile)
            for smile in smile_counts.keys()
        ]
        openff_forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
        openff_topology = openff.toolkit.topology.Topology.from_openmm(
            topology, openff_mols
        )
        openff_topology.box_vectors = [box_size, box_size, box_size] * angstrom
        system = openff_forcefield.create_openmm_system(openff_topology)
        return system

    def return_simulation(self) -> Simulation:
        """
        Uses the settings specified in the constructor to instantiate a OpenMM.Simulation.

        Returns
        -------
        simulation

        """
        system, topology = self._smiles_to_system_topology(
            self.smile_counts, self.density
        )
        integrator = (
            getattr(openmm, self.integrator)
            if self.integrator
            else LangevinMiddleIntegrator
        )
        platform = (
            openmm.Platform.getPlatformByName(self.platform) if self.platform else None
        )
        state = str(self.state)
        simulation = Simulation(topology, system, integrator, platform, state)
        if not state:
            box_size = self._smiles_to_cube_size(self.smile_counts, self.density)
            coordinates = self._smiles_to_coordinates(self.smile_counts, box_size)
            simulation.context.setPositions(coordinates)
        return simulation
