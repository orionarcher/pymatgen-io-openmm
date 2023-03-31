from typing import Optional, Union
from openmm.app import Topology as OpenMMTopology
from openmm.unit import elementary_charge, angstrom
import openff.toolkit as tk
from enum import Enum

from pymatgen.io.openmm.exception_messages.parameterizer_messages import (
    PARAMETERIZER_TYPE_REQUIRED,
    UNEXPECTED_TOPOLOGY_TYPE
)


class ParameterizerAssignment(Enum):
    INFERRED = 1
    EXPLICIT = 2
    DEFAULT = 3


class ParameterizerType(Enum):
    OPENMM_PARAMETERIZER: dict[str,list[str]] = {
        "amber": ["amber14-all.xml"],
        "amoeba": ["amoeba2018.xml","amoeba2018_gk.xml"]
    }
    INTERCHANGE_PARAMETERIZER: dict[str,str] = {
        "sage": ["openff_unconstrained-2.0.0.offxml"]
    }
    DEFAULT_PARAMETERIZER: dict[str,str] = {
    }


class Parameterizer:
    
    """A class modeling the parameterizer object for parameterizing an openmm system"""
    def __init__(
            self,
            topology: Union[tk.Topology,OpenMMTopology],
            mol_specs,
            box,
            force_field,
            parameterizer_type: Optional[ParameterizerType] = None, 
            parameterizer_assignment: ParameterizerAssignment = ParameterizerAssignment.INFERRED
            ):

            self.assign_parameterizer(force_field,parameterizer_type,parameterizer_assignment)
            self.topology = topology
            self.mol_specs = mol_specs
            self.box = box
            self.force_field = force_field

            self.assert_valid_inputs()
             
            

    def infer_parameterizer_type(self, force_field):
        if ParameterizerType.DEFAULT_PARAMETERIZER.value.get(force_field):
            self.parameterizer_type = ParameterizerType.DEFAULT_PARAMETERIZER
        elif ParameterizerType.INTERCHANGE_PARAMETERIZER.value.get(force_field):
            self.parameterizer_type = ParameterizerType.INTERCHANGE_PARAMETERIZER
        elif ParameterizerType.OPENMM_PARAMETERIZER.value.get(force_field):
            self.parameterizer_type = ParameterizerType.OPENMM_PARAMETERIZER
        else:
            self.parameterizer_type = ParameterizerType.DEFAULT_PARAMETERIZER

    def assign_parameterizer(self, force_field: str, parameterizer_type: Optional[ParameterizerType], parameterizer_assignment: ParameterizerAssignment):
        if parameterizer_assignment == ParameterizerAssignment.INFERRED:
            self.infer_parameterizer_type(force_field)
        elif parameterizer_assignment == ParameterizerAssignment.EXPLICIT:
            assert parameterizer_type != None, PARAMETERIZER_TYPE_REQUIRED
            self.parameterizer_type = parameterizer_type
        elif parameterizer_assignment == ParameterizerAssignment.DEFAULT:
            self.parameterizer_type  = ParameterizerType.DEFAULT_PARAMETERIZER
        else:
            self.parameterizer_type  = ParameterizerType.DEFAULT_PARAMETERIZER
    
    def assert_valid_inputs(self):
        assert self.force_field in self.parameterizer_type.value.keys(), f"{self.parameterizer_type.name} only supports the following force fields ${self.parameterizer_type.value.keys()}"

        if self.parameterizer_type == ParameterizerType.INTERCHANGE_PARAMETERIZER:
            assert type(self.topology) == tk.Topology, UNEXPECTED_TOPOLOGY_TYPE(self.parameterizer_type, type(self.topology), tk.Topology)
        if self.parameterizer_type == ParameterizerType.OPENMM_PARAMETERIZER:
            assert type(self.topology) == OpenMMTopology, UNEXPECTED_TOPOLOGY_TYPE(self.parameterizer_type, type(self.topology), OpenMMTopology)
    
    def parameterize_system(self)->any:
        if self.parameterizer_type == ParameterizerType.OPENMM_PARAMETERIZER:
            return self.parameterize_w_openmm()
        elif self.parameterizer_type == ParameterizerType.INTERCHANGE_PARAMETERIZER:
            return self.parameterize_w_interchange()
        else:
            raise NotImplementedError("The only implemented Parameterizers are OPENMM and INTERCHANGE")

    def parameterize_w_interchange(self)->any:
        from openff.interchange import Interchange
        from openff.toolkit import ForceField
        import numpy as np

        box_arr = np.array(self.box)
        box_matrix = np.diag(box_arr[3:6] - box_arr[0:3]) * angstrom
        force_field = ForceField(self.parameterizer_type.value.get(self.force_field))
        
        interchange = Interchange.from_smirnoff(
            force_field=force_field,
            topology=self.topology,
            charge_from_molecules=[spec["openff_mol"] for spec in self.mol_specs],
            box=box_matrix,
            allow_nonintegral_charges=True,
        )
        return interchange.to_openmm()

    def parameterize_w_openmm(self)->any:
        from openmm.app import ForceField, NoCutoff
        from openmm.app.internal.customgbforces import CustomGBForce
        from simtk import openmm as mm
        from openmm.unit import nanometer,angstrom
        
        import numpy as np

        box_arr = np.array(self.box)
        box_matrix = np.diag(box_arr[3:6] - box_arr[0:3]) * angstrom

        force_field = ForceField(*(self.parameterizer_type.value.get(self.force_field)))

        system = force_field.createSystem(topology=self.topology, nonbondedMethod=NoCutoff, polarization='mutual', mutualInducedTargetEpsilon=0.00001,constraints=None, rigidWater=False)

        return system
    
    def generate_residues(self):
        return
        
def generator(forcefield, residue):
    """
    Generate a residue template for a given residue topology using a ForceField object.

    Parameters
    ----------
    forcefield : openmm.app.ForceField
        The ForceField object to which residue templates and/or parameters are to be added.
    residue : openmm.app.Topology.Residue
        The residue topology for which a template is to be generated.

    Returns
    -------
    success : bool
        If the generator is able to successfully parameterize the residue, `True` is returned.
        If the generator cannot parameterize the residue, it should return `False` and not
        modify `forcefield`.
    """
    
    # Attempt to generate the residue template.
    try:
        template = forcefield.getMatchingTemplate(residue)
    except ValueError:
        # If the residue cannot be parameterized, return False.
        return False

    # Register the residue template with the ForceField object.
    forcefield.registerResidueTemplate(template)

    # Return True to indicate that the residue was successfully parameterized.
    return True