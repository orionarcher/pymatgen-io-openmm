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
)
from pymatgen.io.openmm.sets import OpenMMSet
from pymatgen.io.openmm.utils import (
    get_box,
    get_coordinates,
    xyz_to_molecule,
    smiles_to_atom_type_array,
    smiles_to_resname_array,
    get_atom_map,
    parameterize_w_interchange,
    get_openff_topology,
    infer_openff_mol,
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
        force_field: Union[str, Dict[str, str]] = "sage",
        temperature: float = 298,
        step_size: float = 0.001,
        friction_coefficient: int = 1,
        partial_charge_method: str = "am1bcc",
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
            packmol_timeout: the number of seconds to wait for packmol to finish before
                raising an Error.
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
        # old verification and coercion
        # for charge_tuple in self.partial_charges:
        #     if len(charge_tuple) != 2:
        #         raise ValueError(
        #             "partial_charges must be a list of tuples, where the first element is a "
        #             "pymatgen.Molecule or a path to an xyz file and the second element is an "
        #             "array of charges."
        #         )
        # self.partial_charges = list(
        #     map(
        #         lambda charge_tuple: (
        #             xyz_to_molecule(charge_tuple[0]),
        #             charge_tuple[1],
        #         ),
        #         self.partial_charges,
        #     )
        # )
        # self.initial_geometries = {
        #     smile: xyz_to_molecule(geometry) for smile, geometry in self.initial_geometries.items()
        # }
        # TODO: support mol names
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
            "force_field": self.force_field,
            "temperature": self.temperature,
            "step_size": self.step_size,
            "friction_coefficient": self.friction_coefficient,
            "partial_charge_method": self.partial_charge_method,
            "atom_types": atom_types,
            "atom_resnames": atom_resnames,
        }
        return settings_dict

    def get_input_set(
        self,
        input_mol_dicts: List[Dict[str, Union[str, int, float, np.array]]],
        density: Optional[float] = None,
        box: Optional[List[float]] = None,
    ):
        # TODO: should check uniqueness of input_mol_dicts
        mol_specs = []
        for i, mol_dict in enumerate(input_mol_dicts):
            # TODO: put this in a function
            openff_mol = openff.toolkit.topology.Molecule.from_smiles(mol_dict["smile"])

            # add conformer
            if geometries := mol_dict.get("geometries"):
                if not isinstance(geometries, list):
                    geometries = [geometries]
                for geometry in geometries:
                    mol = xyz_to_molecule(geometry)
                    inferred_mol = infer_openff_mol(mol)
                    is_isomorphic, atom_map = get_atom_map(inferred_mol, openff_mol)
                    if not is_isomorphic:
                        raise ValueError(
                            f"An isomorphism cannot be found between smile {mol_dict['smile']} "
                            f"and the provided geometry {geometry}."
                        )
                    new_mol = pymatgen.core.Molecule.from_sites(
                        [mol.sites[i] for i in atom_map.values()]
                    )
                    openff_mol.add_conformer(new_mol.cart_coords * angstrom)
            else:
                atom_map = {i: i for i in range(openff_mol.n_atoms)}
                # TODO document this
                openff_mol.generate_conformers(
                    n_conformers=mol_dict.get("max_conformers") or 1
                )

            # assign partial charges
            if mol_dict.get("partial_charges") is not None:
                partial_charges = mol_dict.get("partial_charges")
                openff_mol.partial_charges = partial_charges[list(atom_map.values())] * elementary_charge  # type: ignore
            else:
                openff_mol.assign_partial_charges(self.partial_charge_method)
            charge_scaling = mol_dict.get("charge_scaling") or 1
            openff_mol.partial_charges = openff_mol.partial_charges * charge_scaling

            # create mol_spec
            mol_spec = dict(
                name=mol_dict.get("name") or mol_dict["smile"],
                forcefield=str.lower(mol_dict.get("force_field") or self.force_field),  # type: ignore
                smile=mol_dict["smile"],
                count=mol_dict["count"],
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

        # note that these variables are all lists of dictionaries
        # TODO: create molgraphs later
        coordinates = get_coordinates(
            openff_counts, box, self.packmol_random_seed, self.packmol_timeout
        )
        openff_topology = get_openff_topology(openff_counts)
        openmm_topology = openff_topology.to_openmm()

        ffs = np.array([mol_spec.get("forcefield") for mol_spec in mol_specs])
        if np.all(ffs == ffs[0]) and ffs[0] in ["sage", "opls"]:
            system = parameterize_w_interchange(
                openff_topology, mol_specs, box, force_field=ffs[0]
            )
        else:
            # system = parameterize_w_openmm_forcefields(mol_specs)
            raise ValueError(
                "All molecules must use the same force field and it must be 'sage' or 'opls'."
            )
        # figure out FF

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
        # instantiate input files and feed to input_set
        topology_input = TopologyInput(openmm_topology)
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
        # TODO: get_input_settings must be refactored
        input_set.settings = self._get_input_settings(mol_specs, box)
        return input_set

    # def get_input_set(  # type: ignore
    #     self,
    #     smiles: Dict[str, int],
    #     mol_dict: Dict[str, Union[str, int, float, tuple]] = None,  # TODO: some shit
    #     density: Optional[float] = None,
    #     box: Optional[List[float]] = None,
    # ) -> InputSet:
    #     """
    #     This executes all the logic to create the input set. It generates coordinates, instantiates
    #     the OpenMM objects, serializes the OpenMM objects, and then returns an InputSet containing
    #     all information needed to generate a simulaiton.
    #
    #     Please note that if the molecules are chiral, then the SMILEs must specify a
    #     particular stereochemistry.
    #
    #     Args:
    #         smiles: keys are smiles and values are number of that molecule to pack
    #         density: the density of the system. density OR box must be given as an argument.
    #         box: list of [xlo, ylo, zlo, xhi, yhi, zhi] with coordinates given in Angstroms.
    #              Density OR box must be given as an argument.
    #
    #     Returns:
    #         an OpenMM.InputSet
    #     """
    #     assert (density is None) ^ (box is None), "Density OR box must be included, but not both."
    #     smiles = {smile: count for smile, count in smiles.items() if count > 0}
    #     # create dynamic openmm objects with internal methods
    #     topology = get_openmm_topology(smiles)
    #     if box is None:
    #         box = get_box(smiles, density)  # type: ignore
    #     coordinates = get_coordinates(
    #         smiles, box, self.packmol_random_seed, self.initial_geometries, self.packmol_timeout
    #     )
    #     smile_strings = list(smiles.keys())
    #     # TODO implement processing
    #     # process input molecules into processed molecules and openff molecules
    #     # parameterize system with just topology, box, and charged_mols ff dict
    #     # system = parameterize_system_2(topology, box, charged_mols, ff_dict)
    #     system, charged_mols = parameterize_system(
    #         topology,
    #         smile_strings,
    #         box,
    #         self.force_field,
    #         self.partial_charge_method,
    #         self.partial_charge_scaling,
    #         self.partial_charges,
    #         return_charged_mols=True,
    #     )
    #     integrator = LangevinMiddleIntegrator(
    #         self.temperature * kelvin,
    #         self.friction_coefficient / picoseconds,
    #         self.step_size * picoseconds,
    #     )
    #     context = Context(system, integrator)
    #     # context.setPositions needs coordinates in nm, but we have them in
    #     # Angstrom from packmol. Convert.
    #     context.setPositions(np.divide(coordinates, 10))
    #     state = context.getState(getPositions=True)
    #     # instantiate input files and feed to input_set
    #     topology_input = TopologyInput(topology)
    #     system_input = SystemInput(system)
    #     integrator_input = IntegratorInput(integrator)
    #     state_input = StateInput(state)
    #     input_set = OpenMMSet(
    #         inputs={
    #             self.topology_file: topology_input,
    #             self.system_file: system_input,
    #             self.integrator_file: integrator_input,
    #             self.state_file: state_input,
    #         },
    #         topology_file=self.topology_file,
    #         system_file=self.system_file,
    #         integrator_file=self.integrator_file,
    #         state_file=self.state_file,
    #     )
    #     input_set.settings = self._get_input_settings(smiles, box, charged_mols)
    #     return input_set
