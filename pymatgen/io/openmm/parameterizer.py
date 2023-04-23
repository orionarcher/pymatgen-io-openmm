from typing import Optional, Union, List
from openmm.app import Topology as OpenMMTopology
from openmm.unit import angstrom
import openff.toolkit as tk
from openff.interchange import Interchange
import numpy as np

import openmm as mm
from enum import Enum

from pymatgen.io.openmm.exception_messages.parameterizer_messages import (
    PARAMETERIZER_TYPE_REQUIRED,
    INVALID_ASSIGNMENT_TYPE,
    INVALID_PARAMETERIZER_TYPE,
    MULTIPLE_PARAMETERIZERS_NOT_SUPPORTED,
    FORCE_FIELD_CUSTOMIZATION_NOT_SUPPORTED,
    MISSING_CUSTOM_FORCE_FIELD_FILES,
    UNEXPECTED_TOPOLOGY_TYPE,
    UNSUPPORTED_FORCE_FIELD_TYPE
)

__author__ = "Anna Weber"
__version__ = "1.0"
__maintainer__ = "Anna Weber"
__email__ = "annaweberdev@gmail.com"
__date__ = "April 2023"

class ParameterizerAssignment(Enum):
    """
    An enumeration class that represents the different types of parameterizer assignments.

    Parameters
    ----------
    INFERRED : int
        Represents an inferred parameterizer assignment.
    EXPLICIT : int
        Represents an explicit parameterizer assignment.
    DEFAULT : int
        Represents a default parameterizer assignment.
    """
    INFERRED = 1
    EXPLICIT = 2
    DEFAULT = 3

class ParameterizerType(Enum):
    """
    An enumeration class that represents the different types of parameterizers and their respective force fields and the files which define them.

    Attributes
    ----------
    OPENMM_PARAMETERIZER : dict[str,list[str]]
        Represents the OpenMM parameterizer with its corresponding list of force field files.
    INTERCHANGE_PARAMETERIZER : dict[str,str]
        Represents the Interchange parameterizer with its corresponding force field file.
    DEFAULT_PARAMETERIZER : dict[str,str]
        Represents the default parameterizer with no parameterization files.
    """
    #note: charm polar may need to have water model set for expected behavior
    OPENMM_PARAMETERIZER: dict[str,list[str]] = {
        "amber": ["amber14-all.xml"],
        "amoeba": ["amoeba2018.xml","amoeba2018_gk.xml"],
        "charmm_polar": ["charmm_polar_2019.xml"],
        "charmm": ["charmm36.xml","charmm36/water.xml"]
    }
    INTERCHANGE_PARAMETERIZER: dict[str,str] = {
        "sage": ["openff_unconstrained-2.0.0.offxml"]
    }
    DEFAULT_PARAMETERIZER: dict[str,str] = OPENMM_PARAMETERIZER


class Parameterizer:
    """
    A class for parameterizing molecular systems using OpenMM or OpenFF Interchange.

    Parameters
    ----------
    topology : openff.toolkit.topology.Topology
        The topology of the molecular system.
    mol_specs : List[Dict[str, Union[openff.toolkit.topology.Molecule, float]]]
        A list of dictionaries, where each dictionary defines a molecule within the system.
        Each dictionary contains an OpenFF Molecule object and a partial charge (float) for the molecule.
    box : List[float]
        A list of six floats defining the box dimensions.
    force_fields : List[str]
        A list of force field names to use for parameterization.
    parameterizer_types : Optional[dict[str, ParameterizerType]]
        A name for the parameterizer to use, If not specified, defaults to using the default
        parameterizer type for each force field.
    parameterizer_assignment : ParameterizerAssignment
        The method of parameterizer assignment to use for the force fields. Options are ParameterizerAssignment.INFERRED,
        ParameterizerAssignment.EXPLICIT, or ParameterizerAssignment.DEFAULT.
    customize_force_field : bool
        Whether or not to use custom force field files for parameterization.
    custom_file_paths : Optional[List[str]]
        A list of file paths to custom force field files. Only used if customize_force_field is True.
    **kwargs
        Additional keyword arguments to pass to OpenMM.

    Attributes
    ----------
    parameterizer_type : ParameterizerType
        The parameterizer type used for the force fields.
    topology : Any
        The topology object for the molecular system.
    mol_specs : List[Dict[str, Union[openff.toolkit.topology.Molecule, float]]]
        A list of dictionaries defining the molecules in the system.
    box : List[float]
        A list of six floats defining the box dimensions.
    force_fields : List[str]
        A list of force field names to use for parameterization.
    additional_args : dict
        Additional keyword arguments to pass to OpenMM.
    using_custom_files : bool
        Whether or not to use custom force field files for parameterization.
    custom_force_field_files : Optional[List[str]]
        A list of file paths to custom force field files. Only used if using_custom_files is True.

    Raises
    ------
    AssertionError
        If the parameterizer assignment type or parameterizer type (if specified) is not of the correct type.
        If multiple parameterizer types are being used for the force fields.
        If an explicit parameterizer type is required but not provided.
        If the force field type is not supported by the parameterizer.
        If the topology type is unexpected for the parameterizer type.
        If custom force field files are being used with a non-OpenMM parameterizer.
        If no custom force field files are provided when using custom force fields.

    References
    ----------
    * OpenMM documentation: https://docs.openmm.org/
    * OpenFF Interchange: https://openff-interchange.readthedocs.io/
    * OpenFF Toolkit: https://open-forcefield-toolkit.readthedocs.io/

    """
    
    def __init__(
            self,
            topology: tk.Topology,
            mol_specs,
            box,
            force_fields: List[str],
            parameterizer_type: Optional[ParameterizerType] = None, 
            parameterizer_assignment: ParameterizerAssignment = ParameterizerAssignment.INFERRED,
            customize_force_field: bool = False,
            custom_file_paths: Optional[List[str]] = None,
            **kwargs
            ):
            """
            Initialize a Parameterizer instance for parameterizing a molecular system.

            Parameters
            ----------
            topology : tk.Topology
                The molecular topology.
            mol_specs : list
                A list of molecule specifications.
            box : list
                A list of the box dimensions.
            force_fields : list of str
                A list of the force field names.
            parameterizer_types : dict of str: ParameterizerType, optional
                A dictionary mapping force field names to their corresponding parameterizer type. If not specified,
                all force fields will be assigned the default parameterizer type.
            parameterizer_assignment : ParameterizerAssignment, optional
                The method used to assign parameterizer types to the force fields. Options are "INFERRED", "EXPLICIT",
                and "DEFAULT". Defaults to "INFERRED".
            customize_force_field : bool, optional
                Whether or not to use custom force field files. Defaults to False.
            custom_file_paths : list of str, optional
                A list of custom force field file paths. Only used if customize_force_field is True.
            **kwargs
                Additional keyword arguments to pass to the OpenMM `ForceField.createSystem` method, if applicable.

            Raises
            ------
            AssertionError
                If the parameterizer type is not of the correct type.
                If multiple parameterizer types are being used for the force fields.
                If an explicit parameterizer type is required but not provided.
                If the force field type is not supported by the parameterizer.
                If the topology type is unexpected for the parameterizer type.
                If custom force field files are being used with a non-OpenMM parameterizer.
                If no custom force field files are provided when using custom force fields.

            Notes
            -----
            The `mol_specs` argument should be a list of dictionaries, where each dictionary contains the following
            Examples
            --------
            >>> parameterizer = Parameterizer(
                    topology,
                    mol_specs=[
                        {
                            "smile": "O",
                            "force_field": "amoeba",
                            "count": 200
                        },
                        {
                            "smile":"[Cl-]",
                            "force_field": "amoeba",
                            "count": 50,
                            "default_charge_method": "mmff94"
                        }
                    ],
                    box=[0, 0, 0, 1, 1, 1],
                    force_fields=["openff-1.0.0.offxml"],
                    parameterizer_assignment=ParameterizerAssignment.EXPLICIT,
                    parameterizer_types={"openff-1.0.0.offxml": ParameterizerType.OPENMM_PARAMETERIZER},
                    customize_force_field=True,
                    custom_file_paths=["custom.xml"]
                )

            """

            self.parameterizer_type = ParameterizerType.DEFAULT_PARAMETERIZER
            self.assign_parameterizer(force_fields,parameterizer_type,parameterizer_assignment)
            self.assign_topology(topology)
            self.mol_specs = mol_specs
            self.box = box
            self.force_fields = force_fields
            self.additional_args = kwargs
            
            
            self.using_custom_files = customize_force_field
            self.custom_force_field_files = custom_file_paths

            self.assert_valid_inputs()
    
    def _get_force_field_file_strings(self):
        """
        Private method that returns a flattened list of force field parameterization file names.

        Returns
        -------
        list[str]
            Flattened list of force field parameterization file names.
        """
        # Retrieve a nested list of force field parameterization file names for each force field type.
        ff_nested_list = [ff_def for ff_def in [self.parameterizer_type.value.get(ff) for ff in self.force_fields]]
        # Flatten the nested list into a single list of force field parameterization file names.
        flat_ff_list = [ff_file for ff_list in ff_nested_list for ff_file in ff_list]
        # Remove duplicate force field parameterization file names.
        return [flat_ff_list[i] for i in range(len(flat_ff_list)) if flat_ff_list.index(flat_ff_list[i])==i]


    def infer_parameterizer_type(self, force_field):
        """
        Method that infers the parameterizer type for a given force field and assigns it to the corresponding attribute.

        Parameters
        ----------
        force_field : str
            The name of the force field.

        Returns
        -------
        None
        """
        # Infer the parameterizer type for the given force field and assign it to the corresponding attribute.
        if ParameterizerType.DEFAULT_PARAMETERIZER.value.get(force_field):
            self.parameterizer_type = ParameterizerType.DEFAULT_PARAMETERIZER
        elif ParameterizerType.INTERCHANGE_PARAMETERIZER.value.get(force_field):
            self.parameterizer_type = ParameterizerType.INTERCHANGE_PARAMETERIZER
        elif ParameterizerType.OPENMM_PARAMETERIZER.value.get(force_field):
            self.parameterizer_type = ParameterizerType.OPENMM_PARAMETERIZER
        else:
            self.parameterizer_type = ParameterizerType.DEFAULT_PARAMETERIZER

    def assign_parameterizer(self, force_fields: List[str], parameterizer_type: Optional[ParameterizerType], parameterizer_assignment: ParameterizerAssignment):
        """
        Method that assigns a parameterizer type for a system with one or more force fields.

        Parameters
        ----------
        force_fields : List[str]
            A list of force field names to assign a parameterizer type to.
        parameterizer_type : Optional[ParameterizerType]
            The parameterizer type to assign to the force fields (if parameterizer_assignment is ParameterizerAssignment.EXPLICIT).
        parameterizer_assignment : ParameterizerAssignment
            The method of parameterizer assignment to use (INFERRED, EXPLICIT, or DEFAULT).

        Raises
        ------
        AssertionError
            If the parameterizer assignment type or parameterizer type (if specified) is not of the correct type.
            If multiple parameterizer types are being used for the force fields.
            If an explicit parameterizer type is required but not provided.

        Returns
        -------
        None
        """
        prev_parameterizer_type = ParameterizerType.DEFAULT_PARAMETERIZER
       
        for i in range(len(force_fields)):
            ff = force_fields[i]

            assert(type(parameterizer_assignment) == ParameterizerAssignment, INVALID_ASSIGNMENT_TYPE)
            assignment = parameterizer_assignment
            
            if parameterizer_type:
                assert(type(parameterizer_type) == ParameterizerType, INVALID_PARAMETERIZER_TYPE)

            if assignment == ParameterizerAssignment.INFERRED:
                self.infer_parameterizer_type(ff)
            elif assignment == ParameterizerAssignment.EXPLICIT:
                assert parameterizer_type != None, PARAMETERIZER_TYPE_REQUIRED
                self.parameterizer_type = parameterizer_type
            elif assignment == ParameterizerAssignment.DEFAULT:
                self.parameterizer_type  = ParameterizerType.DEFAULT_PARAMETERIZER
            else:
                self.parameterizer_type  = ParameterizerType.DEFAULT_PARAMETERIZER

            if i > 0:
                assert(self.parameterizer_type == prev_parameterizer_type, MULTIPLE_PARAMETERIZERS_NOT_SUPPORTED)
    
    def assign_topology(self, topology:  tk.Topology):
        """
        Method that assigns a topology for the parameterizer based on the parameterizer type.

        Parameters
        ----------
        topology : tk.Topology
            The topology to assign.
        
        Returns
        -------
        None
        """
        if(self.parameterizer_type == ParameterizerType.OPENMM_PARAMETERIZER):
            self.topology = topology.to_openmm()
        else:
            self.topology = topology
    
    def assert_valid_inputs(self):
        """
        Method that validates the inputs used to initialize the parameterizer.

        Raises
        ------
        AssertionError
            If the force field type is not supported by the parameterizer.
            If the topology type is unexpected for the parameterizer type.
            If custom force field files are being used with a non-OpenMM parameterizer.
            If no custom force field files are provided when using custom force fields.

        Returns
        -------
        None
        """
        for i in range(len(self.force_fields)):
            ff = self.force_fields[i]
            assert ff in self.parameterizer_type.value.keys(), UNSUPPORTED_FORCE_FIELD_TYPE(self.parameterizer_type)
            if self.parameterizer_type == ParameterizerType.INTERCHANGE_PARAMETERIZER:
                assert type(self.topology) == tk.Topology, UNEXPECTED_TOPOLOGY_TYPE(self.parameterizer_type, type(self.topology), tk.Topology.__name__)
            if self.parameterizer_type == ParameterizerType.OPENMM_PARAMETERIZER:
                assert type(self.topology) == mm.app.Topology, UNEXPECTED_TOPOLOGY_TYPE(self.parameterizer_type, type(self.topology).__name__, mm.app.Topology.__name__)
        if self.using_custom_files:
            assert self.parameterizer_type == ParameterizerType.OPENMM_PARAMETERIZER, FORCE_FIELD_CUSTOMIZATION_NOT_SUPPORTED
            assert self.custom_force_field_files != None, MISSING_CUSTOM_FORCE_FIELD_FILES

    def parameterize_system(self)->mm.System:
        """
        Method that parameterizes the system using the appropriate parameterizer.

        Raises
        ------
        NotImplementedError
            If the parameterizer type is not OPENMM or INTERCHANGE.

        Returns
        -------
        openmm.System
            The parameterized system.
        """
        if self.parameterizer_type == ParameterizerType.OPENMM_PARAMETERIZER:
            return self.parameterize_w_openmm()
        elif self.parameterizer_type == ParameterizerType.INTERCHANGE_PARAMETERIZER:
            return self.parameterize_w_interchange()
        else:
            raise NotImplementedError("The only implemented Parameterizers are OPENMM and INTERCHANGE")

    def parameterize_w_interchange(self)->mm.System:
        """
        Method that parameterizes the system using OpenFF Interchange.

        Returns
        -------
        openmm.System
            The parameterized system as an OpenMM System object.

        References
        ----------
        * OpenFF Interchange: https://openff-interchange.readthedocs.io/
        * OpenFF Toolkit: https://open-forcefield-toolkit.readthedocs.io/
        """
        # Create a box matrix from the box dimensions.
        box_arr = np.array(self.box)
        box_matrix = np.diag(box_arr[3:6] - box_arr[0:3]) * angstrom

        # Load the force field.
        force_field = tk.ForceField(*(self._get_force_field_file_strings()))
        
        # Create an Interchange object and parameterize the system.
        interchange = Interchange.from_smirnoff(
            force_field=force_field,
            topology=self.topology,
            charge_from_molecules=[spec["openff_mol"] for spec in self.mol_specs],
            box=box_matrix,
            allow_nonintegral_charges=True,
        )
        return interchange.to_openmm()

    def parameterize_w_openmm(self)->mm.System:
        """
        Method that parameterizes the system using OpenMM.

        Returns
        -------
        openmm.System
            The parameterized system as an OpenMM System object.

        References
        ----------
        * OpenMM documentation: https://docs.openmm.org/
        """

        # Load the force field.
        force_field = mm.app.ForceField(*(self._get_force_field_file_strings()))

        # If custom force field files are being used, load them.
        if self.using_custom_files:
            force_field.loadFile(*self.custom_force_field_files)

        # Create an OpenMM system object and parameterize the system.
        system = force_field.createSystem(topology=self.topology, nonbondedMethod=mm.app.NoCutoff,constraints=None, rigidWater=False, nonbondedCutoff=1*mm.unit.nanometer,ewaldErrorTolerance=0.00001)

        return system
    