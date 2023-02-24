from pydantic import BaseModel, PositiveInt, confloat, constr, validator
from typing import List, Optional, Union
from pathlib import Path
import pymatgen
import openff.toolkit as tk

from pymatgen.io.openmm.utils import xyz_to_molecule


class Geometry(BaseModel):
    """
    A geometry schema to be used for input to OpenMMSolutionGen.
    """

    xyz: Union[pymatgen.core.Molecule, str, Path]

    @validator("xyz")
    def xyz_is_valid(cls, xyz):
        """check that xyz generates a valid molecule"""
        try:
            xyz_to_molecule(xyz)
        except Exception:
            raise ValueError(f"Invalid xyz file or molecule: {xyz}")
        return xyz_to_molecule(xyz)


class InputMoleculeSpec(BaseModel):
    """
    A molecule schema to be used for input to OpenMMSolutionGen.
    """

    smile: str
    count: int
    name: Optional[str] = None
    charge_scaling: confloat(ge=0.1, le=10) = 1.0  # type: ignore
    force_field: Optional[constr(to_lower=True)] = None  # type: ignore
    geometries: Optional[List[Geometry]] = None
    partial_charges: Optional[List[float]] = None
    max_conformers: PositiveInt = 1

    class Config:
        # needed to allow for np.ndarray
        arbitrary_types_allowed = True

    @validator("smile")
    def smile_is_valid(cls, smile):
        """check that smile generates a valid molecule"""
        try:
            tk.Molecule.from_smiles(smile)
        except Exception:
            raise ValueError(f"Invalid SMILES string: {smile}")
        return smile

    @validator("force_field", pre=True)
    def lower_case_ff(cls, force_field):
        """check that force_field is valid"""
        return force_field.lower()

    @validator("name")
    def assign_name(cls, name, values):
        """assign name if not provided"""
        if name is None:
            return values["smile"]
        return name

    @validator("geometries", pre=True)
    def convert_xyz_to_geometry(cls, geometries):
        """convert xyz to Geometry"""
        if geometries is not None:
            return [Geometry(xyz=xyz) for xyz in geometries]
        return geometries

    @validator("partial_charges", pre=True)
    def check_geometry_is_set(cls, partial_charges, values):
        """check that geometries is set if partial_charges is set"""
        if partial_charges is not None:
            if values.get("geometries") is None:
                raise ValueError("geometries must be set if partial_charges is set")
        return list(partial_charges)
