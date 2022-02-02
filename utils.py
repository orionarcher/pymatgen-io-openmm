"""
Utility functions for OpenMM simulation setup.
"""
from typing import Dict, List, Union, Tuple
import pathlib
from pathlib import Path
import warnings
import tempfile

import numpy as np
from openbabel import pybel
import parmed
import rdkit
import openff
from openff.toolkit.typing.engines import smirnoff
from openff.toolkit.typing.engines.smirnoff.parameters import LibraryChargeHandler
import openmm
from openmm.unit import elementary_charge
from openmm.app import Topology
from openmm.app import ForceField as omm_ForceField
from openmm.app.forcefield import PME
from openmmforcefields.generators import (
    GAFFTemplateGenerator,
    SMIRNOFFTemplateGenerator,
)

import pymatgen
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.xyz import XYZ
from pymatgen.io.packmol import PackmolBoxGen


def smile_to_parmed_structure(smile: str) -> parmed.Structure:
    """
    Convert a SMILE to a Parmed.Structure.
    """
    mol = pybel.readstring("smi", smile)
    mol.addh()
    mol.make3D()
    with tempfile.NamedTemporaryFile() as f:
        mol.write(format="mol", filename=f.name, overwrite=True)
        structure = parmed.load_file(f.name)[0]  # load_file is returning a list for some reason
    return structure


def smile_to_molecule(smile: str) -> pymatgen.core.Molecule:
    """
    Convert a SMILE to a Pymatgen.Molecule.
    """
    mol = pybel.readstring("smi", smile)
    mol.addh()
    mol.make3D()
    adaptor = BabelMolAdaptor(mol.OBMol)
    return adaptor.pymatgen_mol


def get_box(smiles: Dict[str, int], density: float) -> List[float]:
    """
    Calculates the dimensions of a cube necessary to contain the given molecules with
    given density. The corner of the cube is at the origin. Units are angstrom.

    Args:
        smiles: keys are smiles and values are number of that molecule to pack
        density: guessed density of the solution, larger densities will lead to smaller cubes.

    Returns:
        dimensions: array of [0, 0, 0, side_length, side_length, side_length]
    """
    cm3_to_A3 = 1e24
    NA = 6.02214e23
    mols = [smile_to_molecule(smile) for smile in smiles.keys()]
    mol_mw = np.array([mol.composition.weight for mol in mols])
    counts = np.array(list(smiles.values()))
    total_weight = sum(mol_mw * counts)
    box_volume = total_weight * cm3_to_A3 / (NA * density)
    side_length = round(box_volume ** (1 / 3), 2)
    return [0, 0, 0, side_length, side_length, side_length]


def n_mols_from_mass_ratio(n_mol: int, smiles: List[str], mass_ratio: List[float]) -> np.ndarray:
    """
    Calculates the number of mols needed to yield a given mass ratio.

    Args:
        n_mol: total number of mols. returned array will sum to n_mol. e.g. sum(n_mols) = n_mol.
        smiles: a list of smiles. e.g. ["O", "CCO"]
        mass_ratio: mass ratio of smiles. e.g. [9, 1]

    Returns:
        n_mols: number of each smile needed for mass ratio.
    """
    mols = [smile_to_molecule(smile) for smile in smiles]
    mws = np.array([mol.composition.weight for mol in mols])
    mol_ratio = np.array(mass_ratio) / mws
    mol_ratio /= sum(mol_ratio)
    return np.round(mol_ratio * n_mol)


def n_mols_from_volume_ratio(
    n_mol: int, smiles: List[str], volume_ratio: List[float], densities: List[float]
) -> np.ndarray:
    """
    Calculates the number of mols needed to yield a given volume ratio.

    Args:
        n_mol: total number of mols. returned array will sum to n_mol. e.g. sum(n_mols) = n_mol.
        smiles: a list of smiles. e.g. ["O", "CCO"]
        volume_ratio: volume ratio of smiles. e.g. [9, 1]
        densities: density of each smile. e.g. [1, 0.79]

    Returns:
        n_mols: number of each smile needed for volume ratio.

    """
    mass_ratio = np.array(volume_ratio) * np.array(densities)
    return n_mols_from_mass_ratio(n_mol, smiles, mass_ratio)


def n_solute_from_molarity(molarity: float, volume: float) -> int:
    """
    Calculates the number of solute molecules needed for a given molarity.

    Args:
        molarity: molarity of solute desired.
        volume: volume of box in liters.

    Returns:
        n_solute: number of solute molecules

    """
    NA = 6.02214e23
    n_solute = volume * NA * molarity
    return round(n_solute)


def calculate_molarity(volume, n_solute):
    """
    Calculate the molarity of a number of solutes in a volume.
    """
    NA = 6.02214e23
    molarity = n_solute / (volume * NA)
    return molarity


def get_atom_map(inferred_mol, openff_mol) -> Tuple[bool, Dict[int, int]]:
    """
    Get a mapping between two openff Molecules.
    """
    # do not apply formal charge restrictions
    kwargs = dict(
        return_atom_map=True,
        formal_charge_matching=False,
    )
    isomorphic, atom_map = openff.toolkit.topology.Molecule.are_isomorphic(openff_mol, inferred_mol, **kwargs)
    if isomorphic:
        return True, atom_map
    # relax stereochemistry restrictions
    kwargs["atom_stereochemistry_matching"] = False
    kwargs["bond_stereochemistry_matching"] = False
    isomorphic, atom_map = openff.toolkit.topology.Molecule.are_isomorphic(openff_mol, inferred_mol, **kwargs)
    if isomorphic:
        print(f"stereochemistry ignored when matching inferred" f"mol: {openff_mol} to {inferred_mol}")
        return True, atom_map
    # relax bond order restrictions
    kwargs["bond_order_matching"] = False
    isomorphic, atom_map = openff.toolkit.topology.Molecule.are_isomorphic(openff_mol, inferred_mol, **kwargs)
    if isomorphic:
        print(f"stereochemistry ignored when matching inferred" f"mol: {openff_mol} to {inferred_mol}")
        print(f"bond_order restrictions ignored when matching inferred" f"mol: {openff_mol} to {inferred_mol}")
        return True, atom_map
    return False, {}


def infer_openff_mol(mol_geometry: Union[pymatgen.core.Molecule, str, Path]) -> openff.toolkit.topology.Molecule:
    """
    Infer an OpenFF molecule from xyz coordinates.
    """
    if isinstance(mol_geometry, (str, Path)):
        mol_geometry = pymatgen.core.Molecule.from_file(str(mol_geometry))
    with tempfile.NamedTemporaryFile() as f:
        # these next 4 lines are cursed
        pybel_mol = BabelMolAdaptor(mol_geometry).pybel_mol  # pymatgen Molecule
        pybel_mol.write("mol2", filename=f.name, overwrite=True)  # pybel Molecule
        rdmol = rdkit.Chem.MolFromMol2File(f.name, removeHs=False)  # rdkit Molecule
    inferred_mol = openff.toolkit.topology.Molecule.from_rdkit(rdmol, hydrogens_are_explicit=True)  # OpenFF Molecule
    return inferred_mol


def order_molecule_like_smile(smile: str, geometry: Union[pymatgen.core.Molecule, str, Path]):
    """
    Order sites in a pymatgen Molecule to match the canonical ordering generated by rdkit.
    """
    if isinstance(geometry, (str, Path)):
        geometry = pymatgen.core.Molecule.from_file(str(geometry))
    inferred_mol = infer_openff_mol(geometry)
    openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
    is_isomorphic, atom_map = get_atom_map(inferred_mol, openff_mol)
    new_molecule = pymatgen.core.Molecule.from_sites([geometry.sites[i] for i in atom_map.values()])
    return new_molecule


def get_coordinates(
    smiles: Dict[str, int],
    box: List[float],
    random_seed: int = -1,
    smile_geometries: Dict[str, Union[pymatgen.core.Molecule, str, Path]] = None,
) -> np.ndarray:
    """
    Pack the box with the molecules specified by smiles.

    Args:
        smiles: keys are smiles and values are number of that molecule to pack
        box: list of [xlo, ylo, zlo, xhi, yhi, zhi]
        random_seed: the random seed used by packmol
        smile_geometries: a dictionary of smiles and their respective geometries. The
            geometries can be pymatgen Molecules or any file with geometric information
            that can be parsed by OpenBabel.

    Returns:
        array of coordinates for each atom in the box.
    """
    smile_geometries = smile_geometries if smile_geometries else {}
    molecule_geometries = {}
    for smile in smiles.keys():
        if smile in smile_geometries:
            geometry = smile_geometries[smile]
            if isinstance(geometry, (str, Path)):
                geometry = pymatgen.core.Molecule.from_file(geometry)
            molecule_geometries[smile] = order_molecule_like_smile(smile, geometry)
            assert len(geometry) > 0, (
                f"It appears Pymatgen was unable to establish "
                f"an isomorphism between the included geometry "
                f"for {smile} and the molecular graph generated "
                f"by the input file itself. Please ensure you "
                f"included a matching smile and geometry."
            )
        else:
            molecule_geometries[smile] = smile_to_molecule(smile)

    packmol_molecules = []
    for i, smile_count_tuple in enumerate(smiles.items()):
        smile, count = smile_count_tuple
        packmol_molecules.append(
            {
                "name": str(i),
                "number": count,
                "coords": molecule_geometries[smile],
            }
        )
    with tempfile.TemporaryDirectory() as scratch_dir:
        pw = PackmolBoxGen(seed=random_seed).get_input_set(molecules=packmol_molecules, box=box)
        pw.write_input(scratch_dir)
        pw.run(scratch_dir)
        coordinates = XYZ.from_file(pathlib.Path(scratch_dir, "packmol_out.xyz")).as_dataframe()
    raw_coordinates = coordinates.loc[:, "x":"z"].values  # type: ignore
    return raw_coordinates


def get_openmm_topology(smiles: Dict[str, int]) -> openmm.app.Topology:
    """
    Returns an openmm topology with the given SMILEs at the given counts.

    The topology does not contain coordinates.

    Parameters:
        smiles: keys are smiles and values are number of that molecule to pack

    Returns:
        an openmm.app.Topology
    """
    structures = [smile_to_parmed_structure(smile) for smile in smiles.keys()]
    counts = list(smiles.values())
    combined_structs = parmed.Structure()
    for struct, count in zip(structures, counts):
        combined_structs += struct * count
    return combined_structs.topology


def add_mol_charges_to_forcefield(
    forcefield: smirnoff.ForceField,
    charged_openff_mol: List[openff.toolkit.topology.Molecule],
) -> smirnoff.ForceField:
    """
    This is currently depreciated. It may be used in the future.
    """
    charge_type = LibraryChargeHandler.LibraryChargeType.from_molecule(charged_openff_mol)
    forcefield["LibraryCharges"].add_parameter(parameter=charge_type)
    return forcefield


def assign_charges_to_mols(
    smile_strings: List[str],
    partial_charge_method: str,
    partial_charge_scaling: Dict[str, float],
    partial_charges: List[Tuple[Union[pymatgen.core.Molecule, str, Path], np.ndarray]],
):
    """

    This will modify the original force field, not make a copy.

    Args:
        smile_strings: A list of SMILEs strings
        partial_charge_scaling: A dictionary of partial charge scaling for particular species. Keys
        are SMILEs and values are the scaling factor.
        partial_charges: A list of tuples, where the first element of each tuple is a molecular
            geometry and the second element is an array of charges. The geometry can be a
            pymatgen.Molecule or a path to an xyz file. The geometry and charges must have the
            same atom ordering.

    Returns:
       List of charged openff Molecules
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


def assign_small_molecule_ff(molecules: List[openff.toolkit.topology.Molecule], forcefield_name: str):
    """
    Args:
    molecules: List of openff Molecule objects
    forcefield_name: Name of forcefield to apply to the Molecules, can be
                    the absolute path recognized by OpenMM,
                    e.g. "openff-2.0.0" or the generic name, e.g. "sage"

    Returns:
    OpenMM Template
    """
    smirnoff_ff_names = SMIRNOFFTemplateGenerator.INSTALLED_FORCEFIELDS
    gaff_ff_names = GAFFTemplateGenerator.INSTALLED_FORCEFIELDS
    template = None
    if forcefield_name == "sage" or forcefield_name in smirnoff_ff_names:
        ff_name = "openff-2.0.0" if forcefield_name == "sage" else forcefield_name
        template = SMIRNOFFTemplateGenerator(molecules=molecules, forcefield=ff_name)

    elif forcefield_name == "gaff" or forcefield_name in gaff_ff_names:
        ff_name = "gaff-2.11" if forcefield_name == "gaff" else forcefield_name
        template = GAFFTemplateGenerator(molecules=molecules, forcefield=ff_name)
    else:
        raise NotImplementedError(
            f"{forcefield_name} is not supported."
            f"currently only these force fields are supported:"
            f" {' '.join(smirnoff_ff_names+gaff_ff_names)}.\n"
            f"Please select one of the supported force fields."
        )
    return template


def assign_biopolymer_and_water_ff(openmm_forcefield: openmm.app.forcefield, forcefield_assignment: List[str]):
    """

    Args:
    openmm_forcefield: OpenMM forcefield to be updated with forcefield choices
    forcefield_assignment: List of forcefield names to be added to the OpenMM
                        forcefield. Supports casual naming, e.g. "tip3p" or
                        "spce" instead of specigying the full path in OpenMM

    Returns:
    OpenMM Forcefield populated with chosen forcefields
    """
    water_assignment = {
        "amber": {"spce": "amber14/spce.xml", "tip3p": "amber14/tip3p.xml", "tip4p": "amber14/tip4pew.xml"},
        "charmm": {"spce": "charmm36/spce.xml", "tip3p": "charmm36/water.xml", "tip4p": "charmm36/tip4pew.xml"},
    }
    basic_water_ffs = ["tip3p", "spce", "tip4p"]
    biopolymer_ff_category = None
    for ff in forcefield_assignment:
        temp_ff_string = None
        if "amber" in ff.lower():
            temp_ff_string = "amber"
        elif "charmm" in ff.lower():
            temp_ff_string = "charmm"
        if temp_ff_string:
            if biopolymer_ff_category is None:
                biopolymer_ff_category = temp_ff_string
            else:
                if biopolymer_ff_category != temp_ff_string:
                    warnings.warn(
                        f"Did you mean to mix {temp_ff_string} and " f"{biopolymer_ff_category} force fields?"
                    )

    ff_to_load = None
    for ff in forcefield_assignment:
        if ff in basic_water_ffs:
            # Ensure the water model matches the large molecule model
            if biopolymer_ff_category:
                if ff in water_assignment[biopolymer_ff_category].keys():
                    ff_to_load = water_assignment[biopolymer_ff_category][ff]
                else:
                    warnings.warn(f"Did you mean to use {ff} with the " f"" f"{biopolymer_ff_category} force field?")
            # If there isn't a large molecule forcefield required,
            # assume amber14
            else:
                ff_to_load = water_assignment["amber"][ff]
        # TODO: Add lookup to ensure the ff is allowable
        else:
            ff_to_load == ff
        openmm_forcefield.loadFile(ff_to_load)
    return openmm_forcefield


def parameterize_system(
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
        box: list of [xlo, ylo, zlo, xhi, yhi, zhi] in nanometers.
        force_field: Name of the force field or dict of forcefields for each
                    small molecule, e.g. {"O": "spce"}. Small molecule
                    forcefields and water models can either be provided
                    informally, i.e. "gaff" or "sage" for small molecules,
                    or "spce", "tip3p", or "tip4p" for water, or can be
                    formally defined with OpenMM filenames,
                    e.g. "charmm36/water.xml". Large molecule forcefields
                    must be specified with the full path,
                    e.g. "amber14/protein.ff14SB.xml".
        partial_charge_method: Method for OpenFF partial charge assignment
                                for small molecules without charges provided
                                in partial_charges
        partial_charge_scaling: Scaling for partial charges, as a dict
                                of {str: float, . . .}, e.g. {"[Li+]": 0.8}
        partial_charges: List of tuples of (molecule, charges).
                        The Molecule can be a Pymatgen Molecule or the
                        path to a structure file that can be parsed by
                        Pymatgen. Charges is a numpy array with length equal
                         to the number of sites in the molecule.

    Returns:
        an OpenMM.system
    """
    partial_charge_scaling = partial_charge_scaling if partial_charge_scaling else {}
    partial_charges = partial_charges if partial_charges else []
    basic_small_ffs = ["gaff", "sage"]

    all_small_ffs = SMIRNOFFTemplateGenerator.INSTALLED_FORCEFIELDS + GAFFTemplateGenerator.INSTALLED_FORCEFIELDS

    forcefield_omm = omm_ForceField()
    if isinstance(force_field, str):
        ff_name = force_field.lower()
        charged_off_mols = assign_charges_to_mols(
            smile_strings=smile_strings,
            partial_charges=partial_charges,
            partial_charge_scaling=partial_charge_scaling,
            partial_charge_method=partial_charge_method,
        )
        template = assign_small_molecule_ff(molecules=charged_off_mols, forcefield_name=ff_name)
        forcefield_omm.registerTemplateGenerator(template.generator)

    else:
        small_molecules = {}
        biopolymer_or_water = []
        # iterate through each smile, if no forcefielded provided use Sage
        # iterate through each molecule and forcefield input as list
        # Add charges to the molecule if provided
        charged_mols = assign_charges_to_mols(
            smile_strings=smile_strings,
            partial_charges=partial_charges,
            partial_charge_method=partial_charge_method,
            partial_charge_scaling=partial_charge_scaling,
        )
        for smile, charged_mol in zip(smile_strings, charged_mols):
            if smile in force_field.keys():
                ff_name = force_field[smile]
                if ff_name.lower() in all_small_ffs or ff_name.lower() in basic_small_ffs:
                    small_molecules[charged_mol] = ff_name
                else:
                    biopolymer_or_water.append(ff_name)
            else:
                small_molecules[charged_mol] = "sage"
        forcefield_omm = assign_biopolymer_and_water_ff(forcefield_omm, biopolymer_or_water)
        for mol, ff_name in small_molecules.items():
            template = assign_small_molecule_ff(molecules=[mol], forcefield_name=ff_name)
            forcefield_omm.registerTemplateGenerator(template.generator)

    box_size = min(box[3] - box[0], box[4] - box[1], box[5] - box[2])
    nonbondedCutoff = min(10, box_size // 2)
    # TODO: Make insensitive to input units
    periodic_box_vectors = np.array(
        [
            [box[3] - box[0], 0, 0],
            [0, box[4] - box[1], 0],
            [0, 0, box[5] - box[2]],
        ]
    )
    topology.setPeriodicBoxVectors(vectors=periodic_box_vectors)
    system = forcefield_omm.createSystem(topology=topology, nonbondedMethod=PME, nonbondedCutoff=nonbondedCutoff)
    return system
