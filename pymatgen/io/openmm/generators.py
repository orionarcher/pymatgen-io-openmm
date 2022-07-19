"""
Concrete implementations of InputGenerators for the OpenMM IO.
"""

# base python
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple

# scipy
import numpy as np

# openff
import openff

# openmm
from openmm.unit import kelvin, picoseconds
from openmm.openmm import (
    Context,
    LangevinMiddleIntegrator,
)

# pymatgen
import pymatgen.core
from pymatgen.io.core import InputSet, InputGenerator
from pymatgen.io.openmm.inputs import (
    TopologyInput,
    SystemInput,
    IntegratorInput,
    StateInput,
)
from pymatgen.io.openmm.sets import OpenMMSet
from pymatgen.io.openmm.utils import (
    get_box,
    get_coordinates,
    get_openmm_topology,
    parameterize_system,
    xyz_to_molecule,
    smiles_to_atom_type_array,
    smiles_to_resname_array,
)

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


# noinspection PyMethodOverriding
class OpenMMSolutionGen(InputGenerator):
    """
    Generator for an OpenMM InputSet.

    This class is designed for generating simulations of mixed molecular systems. Starting with
    only SMILEs and counts, this class can generate a valid InputSet for a wide variety of
    molecular systems. Currently only the Sage force field is supported.

    This class is only compatible with the Langevin Middle Integrator. To use a different
    integrator, you can first generate the system with OpenMMSolutionGen and then add
    a different integrator to the OpenMMInputSet.
    """

    def __init__(
        self,
        force_field: Union[str, Dict[str, str]] = "Sage",
        temperature: float = 298,
        step_size: float = 0.001,
        friction_coefficient: int = 1,
        partial_charge_method: str = "am1bcc",
        partial_charge_scaling: Optional[Dict[str, float]] = None,
        partial_charges: Optional[List[Tuple[Union[pymatgen.core.Molecule, str, Path], np.ndarray]]] = None,
        initial_geometries: Dict[str, Union[pymatgen.core.Molecule, str, Path]] = None,
        packmol_random_seed: int = -1,
        smile_names: Optional[Dict[str, str]] = None,
        topology_file: Union[str, Path] = "topology.pdb",
        system_file: Union[str, Path] = "system.xml",
        integrator_file: Union[str, Path] = "integrator.xml",
        state_file: Union[str, Path] = "state.xml",
    ):
        """
        Instantiates an OpenMMSolutionGen.

        Args:
            force_field: force field for parameterization, currently supported Force Fields: 'Sage'.
            temperature: the temperature to be added to the integrator (Kelvin).
            step_size: the step size of the simulation (picoseconds).
            friction_coefficient: the friction coefficient which couples the system to
                the heat bath (inverse picoseconds).
            partial_charge_method: Any partial charge method supported by OpenFF Molecule.assign_partial_charges.
                "am1bcc" is recommended for small molecules, "mmff94" is recommended for large molecules.
            partial_charge_scaling: A dictionary of partial charge scaling for particular species. Keys
            are SMILEs and values are the scaling factor.
            partial_charges: A list of tuples, where the first element of each tuple is a molecular
                geometry and the second element is an array of charges. The geometry can be a
                pymatgen.Molecule or a path to an xyz file. The geometry and charges must have the
                same atom ordering.
            initial_geometries: A dictionary where the keys are smiles and the values are
                pymatgen.Molecules or a path to a file with xyz information.
            smile_names: A dictionary of smiles and common names.
            packmol_random_seed: The random seed for Packmol. If -1, a random seed will be generated.
            topology_file: Location to save the Topology PDB.
            system_file: Location to save the System xml.
            integrator_file: Location to save the Integrator xml.
            state_file: Location to save the State xml.
        """
        self.force_field = force_field
        self.temperature = temperature
        self.step_size = step_size
        self.friction_coefficient = friction_coefficient
        self.partial_charge_method = partial_charge_method
        self.partial_charge_scaling = partial_charge_scaling or {}
        self.partial_charges = partial_charges or []
        for charge_tuple in self.partial_charges:
            if len(charge_tuple) != 2:
                raise ValueError(
                    "partial_charges must be a list of tuples, where the first element is a "
                    "pymatgen.Molecule or a path to an xyz file and the second element is an "
                    "array of charges."
                )
        self.partial_charges = list(
            map(
                lambda charge_tuple: (
                    xyz_to_molecule(charge_tuple[0]),
                    charge_tuple[1],
                ),
                self.partial_charges,
            )
        )
        self.initial_geometries = initial_geometries or {}
        self.initial_geometries = {
            smile: xyz_to_molecule(geometry) for smile, geometry in self.initial_geometries.items()
        }
        self.smile_names = smile_names or {}
        self.packmol_random_seed = packmol_random_seed
        self.topology_file = topology_file
        self.system_file = system_file
        self.integrator_file = integrator_file
        self.state_file = state_file

    def _get_input_settings(
        self,
        smiles: Dict[str, int],
        box: List[float],
        charged_mols: List[openff.toolkit.topology.Molecule],
    ) -> Dict:
        atom_types = smiles_to_atom_type_array(smiles)
        atom_resnames = smiles_to_resname_array(smiles, self.smile_names)
        initial_geometries = {smile: geo for smile, geo in self.initial_geometries.items() if smile in smiles}
        partial_charges = {mol.to_smiles(): mol.partial_charges._value for mol in charged_mols}
        settings_dict = {
            "smile_counts": smiles,
            "box": box,
            "force_field": self.force_field,
            "temperature": self.temperature,
            "step_size": self.step_size,
            "friction_coefficient": self.friction_coefficient,
            "partial_charge_method": self.partial_charge_method,
            "partial_charge_scaling": self.partial_charge_scaling,
            "partial_charges": partial_charges,
            "initial_geometries": initial_geometries,
            "smile_names": self.smile_names,
            "atom_types": atom_types,
            "atom_resnames": atom_resnames,
        }
        return settings_dict

    def get_input_set(  # type: ignore
        self,
        smiles: Dict[str, int],
        density: Optional[float] = None,
        box: Optional[List[float]] = None,
    ) -> InputSet:
        """
        This executes all of the logic to create the input set. It generates coordinates, instantiates
        the OpenMM objects, serializes the OpenMM objects, and then returns an InputSet containing
        all information needed to generate a simulaiton.

        Please note that if the molecules are chiral, then the SMILEs must specify a
        particular stereochemistry.

        Args:
            smiles: keys are smiles and values are number of that molecule to pack
            density: the density of the system. density OR box must be given as an argument.
            box: list of [xlo, ylo, zlo, xhi, yhi, zhi] with coordinates given in Angstroms. Density OR box must be given as an argument.

        Returns:
            an OpenMM.InputSet
        """
        assert (density is None) ^ (box is None), "Density OR box must be included, but not both."
        smiles = {smile: count for smile, count in smiles.items() if count > 0}
        # create dynamic openmm objects with internal methods
        topology = get_openmm_topology(smiles)
        if box is None:
            box = get_box(smiles, density)  # type: ignore
        coordinates = get_coordinates(smiles, box, self.packmol_random_seed, self.initial_geometries)
        smile_strings = list(smiles.keys())
        system, charged_mols = parameterize_system(
            topology,
            smile_strings,
            box,
            self.force_field,
            self.partial_charge_method,
            self.partial_charge_scaling,
            self.partial_charges,
            return_charged_mols=True,
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
        input_set.settings = self._get_input_settings(smiles, box, charged_mols)
        return input_set
