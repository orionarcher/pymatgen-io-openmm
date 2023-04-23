"""
Concrete implementations of InputGenerators for the OpenMM IO.
"""

# base python
from pathlib import Path
from typing import Dict, List, Optional, Union


# scipy
import numpy as np

# openff
import openff
import openff.toolkit as tk

# openmm
from openmm.unit import kelvin, picoseconds, angstrom, elementary_charge
from openmm.openmm import (
    Context,
    LangevinMiddleIntegrator,
)

# pymatgen
import pymatgen.core
from pymatgen.io.core import InputGenerator
from pymatgen.io.openmm.inputs import (
    TopologyInput,
    SystemInput,
    IntegratorInput,
    StateInput,
    MSONableInput,
)
from pymatgen.io.openmm.sets import OpenMMSet, OpenMMAlchemySet
from pymatgen.io.openmm.parameterizer import Parameterizer, ParameterizerAssignment, ParameterizerType
from pymatgen.io.openmm.schema import InputMoleculeSpec
from pymatgen.io.openmm.utils import (
    get_box,
    get_coordinates,
    smiles_to_atom_type_array,
    smiles_to_resname_array,
    get_atom_map,
    get_openff_topology,
    infer_openff_mol,
)
from pymatgen.io.openmm.alchemy_utils import AlchemicalReaction, ReactiveSystem

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


# TODO: change to Dataclass
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
        default_force_field: Union[str, Dict[str, str]] = "sage",
        temperature: float = 298,
        step_size: float = 0.001,
        friction_coefficient: int = 1,
        default_charge_method: str = "am1bcc",
        packmol_random_seed: int = -1,
        packmol_timeout: int = 30,
        topology_file: Union[str, Path] = "topology.pdb",
        system_file: Union[str, Path] = "system.xml",
        integrator_file: Union[str, Path] = "integrator.xml",
        state_file: Union[str, Path] = "state.xml",
    ):
        """
        Instantiates an OpenMMSolutionGen.

        Args:
            default_force_field: force field for parameterization, currently supported Force Fields: 'Sage'.
            temperature: the temperature to be added to the integrator (Kelvin).
            step_size: the step size of the simulation (picoseconds).
            friction_coefficient: the friction coefficient which couples the system to
                the heat bath (inverse picoseconds).
            default_charge_method: Any partial charge method supported by OpenFF Molecule.assign_partial_charges.
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
            packmol_timeout: the number of seconds to wait for packmol to finish before
                raising an Error.
            topology_file: Location to save the Topology PDB.
            system_file: Location to save the System xml.
            integrator_file: Location to save the Integrator xml.
            state_file: Location to save the State xml.
        """
        self.default_force_field = default_force_field
        self.temperature = temperature
        self.step_size = step_size
        self.friction_coefficient = friction_coefficient
        self.default_charge_method = default_charge_method
        self.packmol_random_seed = packmol_random_seed
        self.packmol_timeout = packmol_timeout
        self.topology_file = topology_file
        self.system_file = system_file
        self.integrator_file = integrator_file
        self.state_file = state_file

    def _get_input_settings(
        self,
        mol_specs: List[Dict[str, Union[str, int, tk.Molecule]]],
        box: List[float],
    ) -> Dict:
        openff_counts = {spec["openff_mol"]: spec["count"] for spec in mol_specs}
        atom_types = smiles_to_atom_type_array(openff_counts)
        atom_resnames = smiles_to_resname_array(mol_specs)
        # TODO: add mol_specs
        settings_dict = {
            "box": box,
            "force_field": self.default_force_field,
            "temperature": self.temperature,
            "step_size": self.step_size,
            "friction_coefficient": self.friction_coefficient,
            "partial_charge_method": self.default_charge_method,
            "atom_types": atom_types,
            "atom_resnames": atom_resnames,
        }
        # TODO: will need to serialize this to JSON and include in settings
        # ideally the settings should be a pydantic BaseModel
        return settings_dict

    def get_input_set(
        self,
        input_mol_dicts: List[Union[Dict, InputMoleculeSpec]],
        density: Optional[float] = None,
        box: Optional[List[float]] = None,
        parameterizer_type: Optional[ParameterizerType] = None, 
        parameterizer_assignment: ParameterizerAssignment = ParameterizerAssignment.INFERRED,
        customize_force_field: bool = False,
        custom_file_paths: Optional[List[str]] = None,
    ):
        """
        Helper function for instantiating OpenMMSet objects from by specifying a
        list of dictionaries or InputMoleculeSpecs, simulation density or simulation
        dimensions.

        Parameters
        ----------
        input_mol_dicts : List[Union[Dict, InputMoleculeSpec]]
            List of dicts or InputMoleculeSpecs.
            Dicts must have smile and count str keys - see required args for InputMoleculeSpec.
        density : Optional[float]
            simulation density, molecules/atoms per cubic centimeter
        box : Optional[List[float]]
            simulation box dimensions in centimeters
        parameterizer_type : ParameterizerType
            The parameterizer type used for the force fields.
        parameterizer_assignment : ParameterizerAssignment
            How the parameterizer should be assigned for building the system
        using_custom_files : bool
            Whether or not to use custom force field files for parameterization.
        custom_force_field_files : Optional[List[str]]
            A list of file paths to custom force field files. Only used if using_custom_files is True.
        Returns
        -------
        input_set : OpenMMSet
            OpenMM instance with containing simulation input files
        """
        # TODO: add default for density, maybe 1.5?
        # coerce all input_mol_dicts to InputMoleculeSpec
        input_mol_specs = []
        for mol_dict in input_mol_dicts:
            if isinstance(mol_dict, dict):
                input_mol_specs.append(InputMoleculeSpec(**mol_dict))
            else:
                input_mol_specs.append(mol_dict)

        # assert uniqueness of smiles in mol_specs
        smiles = [mol_spec.smile for mol_spec in input_mol_specs]
        if len(smiles) != len(set(smiles)):
            raise ValueError("Smiles in input mol dicts must be unique.")

        mol_specs = []
        for i, mol_dict in enumerate(input_mol_specs):
            # TODO: put this in a function
            openff_mol = openff.toolkit.topology.Molecule.from_smiles(mol_dict.smile)

            # add conformer
            if geometries := mol_dict.geometries:
                for geometry in geometries:
                    inferred_mol = infer_openff_mol(geometry.xyz)
                    is_isomorphic, atom_map = get_atom_map(inferred_mol, openff_mol)
                    if not is_isomorphic:
                        raise ValueError(
                            f"An isomorphism cannot be found between smile {mol_dict.smile} "
                            f"and the provided geometry {geometry.xyz}."
                        )
                    new_mol = pymatgen.core.Molecule.from_sites(
                        [geometry.xyz.sites[i] for i in atom_map.values()]
                    )
                    openff_mol.add_conformer(new_mol.cart_coords * angstrom)
            else:
                atom_map = {i: i for i in range(openff_mol.n_atoms)}
                # TODO document this
                openff_mol.generate_conformers(
                    n_conformers=mol_dict.max_conformers or 1
                )

            # assign partial charges
            if mol_dict.partial_charges is not None:
                partial_charges = np.array(mol_dict.partial_charges)
                openff_mol.partial_charges = partial_charges[list(atom_map.values())] * elementary_charge  # type: ignore
            elif openff_mol.n_atoms == 1:
                openff_mol.partial_charges = (
                    np.array([openff_mol.total_charge.magnitude]) * elementary_charge
                )
            else:
                openff_mol.assign_partial_charges(self.default_charge_method)
            charge_scaling = mol_dict.charge_scaling or 1
            openff_mol.partial_charges = openff_mol.partial_charges * charge_scaling

            # create mol_spec
            mol_spec = dict(
                name=mol_dict.name,
                count=mol_dict.count,
                smile=mol_dict.smile,
                forcefield=mol_dict.force_field or self.default_force_field,  # type: ignore
                formal_charge=int(
                    np.sum(openff_mol.partial_charges.magnitude) / charge_scaling
                ),
                openff_mol=openff_mol,
            )
            mol_specs.append(mol_spec)

        openff_counts = {spec["openff_mol"]: spec["count"] for spec in mol_specs}

        assert (density is None) ^ (
            box is None
        ), "Density OR box must be included, but not both."
        if box is None:
            box = get_box(openff_counts, density)  # type: ignore

        # TODO: create molgraphs later
        coordinates = get_coordinates(
            openff_counts, box, self.packmol_random_seed, self.packmol_timeout
        )
        openff_topology = get_openff_topology(openff_counts)

        ffs = ([mol_spec["forcefield"] for mol_spec in mol_specs])

        parameterizer = Parameterizer(openff_topology, mol_specs,box,ffs, parameterizer_type, parameterizer_assignment, customize_force_field, custom_file_paths)

        system = parameterizer.parameterize_system()
        # TODO: wrap system creation in try/except to catch periodic boundary errors

        integrator = LangevinMiddleIntegrator(
            self.temperature * kelvin,
            self.friction_coefficient / picoseconds,
            self.step_size * picoseconds,
        )
        context = Context(system, integrator)
        # context.setPositions needs coordinates in nm, but we have them in
        # Angstrom from packmol. Convert.
        context.setPositions(np.divide(coordinates, 10))
        state = context.getState(getPositions=True)
        input_set = OpenMMSet(
            inputs={
                self.topology_file: TopologyInput(openff_topology.to_openmm()),
                self.system_file: SystemInput(system),
                self.integrator_file: IntegratorInput(integrator),
                self.state_file: StateInput(state),
            },
            topology_file=self.topology_file,
            system_file=self.system_file,
            integrator_file=self.integrator_file,
            state_file=self.state_file,
        )
        # TODO: get_input_settings must be refactored
        input_set.settings = self._get_input_settings(mol_specs, box)
        return input_set


class OpenMMAlchemyGen(OpenMMSolutionGen):
    """
    An InputGenerator for openmm alchemy.
    """

    def __init__(self, reactive_system_file="reactive_system.json", **kwargs):
        super().__init__(**kwargs)
        self.reactive_system_file = reactive_system_file

    def get_input_set(  # type: ignore
        self,
        input_mol_dicts: List[Union[Dict, InputMoleculeSpec]],
        reactions: List[AlchemicalReaction] = None,
        density: Optional[float] = None,
        box: Optional[List[float]] = None,
    ) -> OpenMMAlchemySet:
        """
        This executes all the logic to create the input set. It generates coordinates, instantiates
        the OpenMM objects, serializes the OpenMM objects, and then returns an InputSet containing
        all information needed to generate a simulation. In addition, it also identifies what atoms
        will participate in a given AlchemicalReaction and includes that information in the reaction_spec.

        Please note that if the molecules are chiral, then the SMILEs must specify a
        particular stereochemistry.

        Args:
            input_mol_dicts: a set of input_mol_dicts.
            reactions: a list of AlchemicalReactions specifying the reactions to perform.
            density: the density of the system. density OR box must be given as an argument.
            box: list of [xlo, ylo, zlo, xhi, yhi, zhi]. density OR box must be given as an argument.

        Returns:
            an OpenMM.InputSet
        """
        reactions = reactions or []

        input_set = super().get_input_set(input_mol_dicts, density, box)
        openff_counts = {
            tk.Molecule.from_smiles(mol_dict["smile"]): mol_dict["count"]
            for mol_dict in input_mol_dicts
        }
        reactive_system = ReactiveSystem.from_reactions(openff_counts, reactions)
        rxn_input_set = OpenMMAlchemySet(
            inputs={
                **input_set.inputs,
                self.reactive_system_file: MSONableInput(reactive_system),
            },
            topology_file=self.topology_file,
            system_file=self.system_file,
            integrator_file=self.integrator_file,
            state_file=self.state_file,
            reactive_system_file=self.reactive_system_file,
        )
        return rxn_input_set
