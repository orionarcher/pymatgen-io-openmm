"""
Concrete implementations of InputGenerators for the OpenMM IO.
"""

# base python
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple

# scipy
import numpy as np

# openff

# openmm
from openmm.unit import kelvin, picoseconds
from openmm import (
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
        self.partial_charge_scaling = partial_charge_scaling if partial_charge_scaling else {}
        self.partial_charges = partial_charges if partial_charges else []
        self.initial_geometries = initial_geometries if initial_geometries else {}
        self.packmol_random_seed = packmol_random_seed
        self.topology_file = topology_file
        self.system_file = system_file
        self.integrator_file = integrator_file
        self.state_file = state_file

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
            box: list of [xlo, ylo, zlo, xhi, yhi, zhi]. density OR box must be given as an argument.

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
        system = parameterize_system(
            topology,
            smile_strings,
            box,
            self.force_field,
            self.partial_charge_method,
            self.partial_charge_scaling,
            self.partial_charges,
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
