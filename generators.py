"""
Concrete implementations of InputGenerators for the OpenMM IO.
"""

# base python
from pathlib import Path
import warnings
from typing import Union, Optional, Dict, List, Tuple

# scipy
import numpy as np

# openff
import openff
import openff.toolkit
from openff.toolkit.typing.engines import smirnoff
from openff.toolkit.typing.engines.smirnoff.parameters import LibraryChargeHandler

# openmm
import openmm
from openmm.unit import kelvin, picoseconds, elementary_charge, angstrom
from openmm.app import Topology
from openmm.app import ForceField as omm_ForceField
from openmm import (
    Context,
    LangevinMiddleIntegrator,
)
from openmm.app.forcefield import PME
from openmmforcefields.generators import (
    GAFFTemplateGenerator,
    SMIRNOFFTemplateGenerator,
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
    get_atom_map,
    infer_openff_mol,
    get_coordinates,
    get_openmm_topology,
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
        system = self._parameterize_system(
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

    @staticmethod
    def _add_mol_charges_to_forcefield(
        forcefield: smirnoff.ForceField,
        charged_openff_mol: List[openff.toolkit.topology.Molecule],
    ) -> smirnoff.ForceField:

        charge_type = LibraryChargeHandler.LibraryChargeType.from_molecule(charged_openff_mol)
        forcefield["LibraryCharges"].add_parameter(parameter=charge_type)
        return forcefield

    @staticmethod
    def _assign_charges_to_mols(
        smile_strings: List[str],
        partial_charge_method: str,
        partial_charge_scaling: Dict[str, float],
        partial_charges: List[Tuple[Union[pymatgen.core.Molecule, str, Path], np.ndarray]],
    ):
        """

        This will modify the original force field, not make a copy.

        Args:
            forcefield: force field that will have partial charges added.
            partial_charge_scaling: A dictionary of partial charge scaling for particular species. Keys
            are SMILEs and values are the scaling factor.
            partial_charges: A list of tuples, where the first element of each tuple is a molecular
                geometry and the second element is an array of charges. The geometry can be a
                pymatgen.Molecule or a path to an xyz file. The geometry and charges must have the
                same atom ordering.

        Returns:
            forcefield with partial charges added.
        """
        # loop through partial charges to add to force field
        matched_mols = set()
        inferred_mols = set()
        charged_mols = []
        for smile in smile_strings:
            # detect charge scaling, set scaling parameter
            if smile in partial_charge_scaling.keys():
                charge_scaling = partial_charge_scaling[smile]
            else:
                charge_scaling = 1
            openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
            # assign charges from isomorphic charges, if they exist
            is_isomorphic = False
            for mol_xyz, charges in partial_charges:
                inferred_mol = infer_openff_mol(mol_xyz)
                inferred_mols.add(inferred_mol)
                is_isomorphic, atom_map = get_atom_map(inferred_mol, openff_mol)
                # if is_isomorphic to a mol_xyz in the system, add to openff_mol else, warn user
                if is_isomorphic:
                    reordered_charges = np.array([charges[atom_map[i]] for i, _ in enumerate(charges)])
                    openff_mol.partial_charges = reordered_charges * charge_scaling * elementary_charge
                    matched_mols.add(inferred_mol)
                    break
            if not is_isomorphic:
                # assign partial charges if there was no match
                openff_mol.assign_partial_charges(partial_charge_method)
                openff_mol.partial_charges = openff_mol.partial_charges * charge_scaling
            # finally, add charged mol to force_field
            charged_mols.append(openff_mol)
            # return a warning if some partial charges were not matched to any mol_xyz
        for unmatched_mol in inferred_mols - matched_mols:
            warnings.warn(f"{unmatched_mol} in partial_charges is not isomorphic to any SMILE in the system.")
        return charged_mols

    @staticmethod
    def _parameterize_system(
        topology: Topology,
        smile_strings: List[str],
        box: List[float],
        force_field: Union[str, Dict[str, str]] = "sage",
        partial_charge_method: str = "am1bcc",
        partial_charge_scaling: Dict[str, float] = None,
        partial_charges: List[Tuple[Union[pymatgen.core.Molecule, str, Path], np.ndarray]] = [],
    ) -> openmm.System:
        """
        Parameterize an OpenMM system.

        Args:
            topology: an OpenMM topology.
            smile_strings: a list of SMILEs representing each molecule in the system.
            box: list of [xlo, ylo, zlo, xhi, yhi, zhi].
            force_field: name of the force field. Currently only Sage is supported.

        Returns:
            an OpenMM.system
        """

        partial_charge_scaling = partial_charge_scaling if partial_charge_scaling else {}
        partial_charges = partial_charges if partial_charges else []
        supported_force_fields = ["Sage"]
        if isinstance(force_field, str):
            if force_field.lower() == "sage":
                openff_forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
                charged_openff_mols = OpenMMSolutionGen._assign_charges_to_mols(
                    smile_strings,
                    partial_charge_method,
                    partial_charge_scaling,
                    partial_charges,
                )
                openff_topology = openff.toolkit.topology.Topology.from_openmm(topology, charged_openff_mols)
                box_vectors = list(np.array(box[3:6]) - np.array(box[0:3])) * angstrom
                openff_topology.box_vectors = box_vectors
                system = openff_forcefield.create_openmm_system(
                    openff_topology,
                    charge_from_molecules=charged_openff_mols,
                    allow_nonintegral_charges=True,
                )
                return system
        else:
            # TODO: Make decisions for user about ff name
            # TODO: Dict instead of list of tuples
            # TODO: Make periodic
            small_ffs = [
                "smirnoff99Frosst-1.0.2",
                "smirnoff99Frosst-1.0.0",
                "smirnoff99Frosst-1.1.0",
                "smirnoff99Frosst-1.0.4",
                "smirnoff99Frosst-1.0.8",
                "smirnoff99Frosst-1.0.6",
                "smirnoff99Frosst-1.0.3",
                "smirnoff99Frosst-1.0.1",
                "smirnoff99Frosst-1.0.5",
                "smirnoff99Frosst-1.0.9",
                "smirnoff99Frosst-1.0.7",
                "openff-1.0.1",
                "openff-1.1.1",
                "openff-1.0.0-RC1",
                "openff-1.2.0",
                "openff-1.3.0",
                "openff-2.0.0-rc.2",
                "openff-2.0.0",
                "openff-1.1.0",
                "openff-1.0.0",
                "openff-1.0.0-RC2",
                "openff-1.3.1",
                "openff-1.2.1",
                "openff-1.3.1-alpha.1",
                "openff-2.0.0-rc.1",
                "gaff-1.4",
                "gaff-1.8",
                "gaff-1.81",
                "gaff-2.1",
                "gaff-2.11",
            ]
            small_molecules = {}
            large_or_water = {}
            # iterate through each molecule and forcefield input as list
            for smile, ff_name in force_field.items():
                openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
                # Assign mols and forcefield as small molecule vs AMBER or
                # CHARMM
                if ff_name.lower() in small_ffs:
                    small_molecules[openff_mol] = ff_name.lower()
                else:
                    large_or_water[openff_mol] = ff_name.lower()
            forcefield_omm = omm_ForceField()
            for ff in large_or_water.values():
                forcefield_omm.loadFile(ff)
            for mol, ff in small_molecules.items():
                if "gaff" in ff:
                    gaff = GAFFTemplateGenerator(molecules=mol, forcefield=ff)
                    forcefield_omm.registerTemplateGenerator(gaff.generator)
                elif "smirnoff" in ff or "openff" in ff:
                    sage = SMIRNOFFTemplateGenerator(molecules=mol, forcefield=ff)
                    forcefield_omm.registerTemplateGenerator(sage.generator())
            box_size = min(box[3] - box[0], box[4] - box[1], box[5] - box[2])
            nonbondedCutoff = min(10, box_size // 2)
            periodic_box_vectors = np.multiply(
                np.array(
                    [
                        [box[3] - box[0], 0, 0],
                        [0, box[4] - box[1], 0],
                        [0, 0, box[5] - box[2]],
                    ]
                ),
                0.1,
            )  # needs to be nanometers, assumes box in angstroms
            topology.setPeriodicBoxVectors(vectors=periodic_box_vectors)
            system = forcefield_omm.createSystem(
                topology=topology, nonbondedMethod=PME, nonbondedCutoff=nonbondedCutoff
            )
            return system
        raise NotImplementedError(
            f"currently only these force fields are supported: {' '.join(supported_force_fields)}.\n"
            f"Please select one of the supported force fields."
        )
