"""
Concrete implementations of InputFiles, InputSet, and InputGenerator for the OpenMM IO.
"""

# base python
import io
from pathlib import Path
import pathlib
import tempfile
import warnings
from typing import Union, Optional, Dict, List, Tuple

# cheminformatics
import numpy as np
import rdkit
from openbabel import pybel
import parmed

# openff
import openff
import openff.toolkit
from openff.toolkit.typing.engines import smirnoff
from openff.toolkit.typing.engines.smirnoff.parameters import LibraryChargeHandler

# openmm
import openmm
from openmm.unit import kelvin, picoseconds, elementary_charge, angstrom
from openmm.app import Simulation, PDBFile, Topology
from openmm import (
    XmlSerializer,
    System,
    Integrator,
    State,
    Context,
    LangevinMiddleIntegrator,
)

# pymatgen
import pymatgen.core
from pymatgen.io.core import InputFile, InputSet, InputGenerator
from pymatgen.io.packmol import PackmolBoxGen
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.io.xyz import XYZ

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


class TopologyInput(InputFile):
    """
    Input handler for OpenMM topologies. Stores and parses PDB files.
    """

    def __init__(self, topology: Topology, positions: Optional[Union[List, np.ndarray]] = None):
        self.content = self._serialize(topology, positions)

    @staticmethod
    def _serialize(topology, positions) -> str:
        if not positions:
            positions = np.zeros(shape=(topology.getNumAtoms(), 3))
        with io.StringIO() as s:
            PDBFile.writeFile(topology, positions, file=s)
            s.seek(0)
            pdb = s.read()
        return pdb

    def get_string(self) -> str:
        return self.content

    @classmethod
    def from_string(cls, contents: str) -> InputFile:
        with io.StringIO(contents) as s:
            pdb = PDBFile(s)
            topology = pdb.getTopology()
        return TopologyInput(topology)

    def get_topology(self) -> Topology:
        """
        Returns the OpenMM topology represented by the TopologyInput.

        Returns:
            openmm.app.Topology
        """
        with io.StringIO(self.content) as s:
            pdb = PDBFile(s)
            topology = pdb.getTopology()
        return topology


class XmlInput(InputFile):
    """
    A standardized definition for InputFiles based on OpenMM XML serialization.

    compatible with any OpenMM object with a serialization proxy registered:
    https://github.com/openmm/openmm/blob/master/serialization/src/SerializationProxyRegistration.cpp
    """

    def __init__(self, openmm_object):
        self.content = self._serialize(openmm_object)

    @staticmethod
    def _serialize(openmm_object) -> str:
        return XmlSerializer.serialize(openmm_object)

    def get_string(self) -> str:
        return self.content

    @classmethod
    def from_string(cls, contents: str):
        """
        This is a template that should be overwritten. Replace XmlInput with child class.
        """
        return XmlInput(XmlSerializer.deserialize(contents))


class SystemInput(XmlInput):
    """
    Input handler for OpenMM systems. Stores and parses XML files.
    """

    @classmethod
    def from_string(cls, contents: str) -> InputFile:
        return SystemInput(XmlSerializer.deserialize(contents))

    def get_system(self) -> System:
        """
        Returns the OpenMM system represented by the SystemInput.

        Returns:
            openmm.System
        """
        return XmlSerializer.deserialize(self.content)


class IntegratorInput(XmlInput):
    """
    Input handler for OpenMM integrators. Stores and parses XML files.
    """

    @classmethod
    def from_string(cls, contents: str) -> InputFile:
        return IntegratorInput(XmlSerializer.deserialize(contents))

    def get_integrator(self) -> Integrator:
        """
        Returns the OpenMM integrator represented by the IntegratorInput.

        Returns:
            openmm.Integrator
        """
        return XmlSerializer.deserialize(self.content)


class StateInput(XmlInput):
    """
    State handler for OpenMM integrators. Stores and parses XML files.
    """

    @classmethod
    def from_string(cls, contents: str) -> InputFile:
        return StateInput(XmlSerializer.deserialize(contents))

    def get_state(self) -> State:
        """
        Returns the OpenMM state represented by the StateInput.

        Returns:
            openmm.State
        """
        return XmlSerializer.deserialize(self.content)


class OpenMMSet(InputSet):
    """
    An InputSet for OpenMM Simulations.

    This is a container for the serialized topology, system, integrator, and state files.
    Topology is stored as a pdb and the other files are stored as xml. The state file is
    optional but positions must be manually assigned if no state is provided.
    """

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        topology_file: str = "topology.pdb",
        system_file: str = "system.xml",
        integrator_file: str = "integrator.xml",
        state_file: str = "state.xml",
    ):
        """
        Instantiates an InputSet from a directory containing a Topology PDB, a
        System XML, a Integrator XML, and (optionally) a State XML. If no State
        is given, system coordinates must be manually assigned.

        Args:
            directory: directory that holds the input files.
            topology_file: name of the pdb file with topological information.
            system_file: name of the serialized System xml file.
            integrator_file: name of the serialized Integrator xml file.
            state_file: name of the serialized State xml file. If there is no state_file,
                then positions must be set for simulation.

        Returns:
            an OpenMMSet
        """
        source_dir = Path(directory)
        topology_input = TopologyInput.from_file(source_dir / topology_file)
        system_input = SystemInput.from_file(source_dir / system_file)
        integrator_input = IntegratorInput.from_file(source_dir / integrator_file)
        inputs = {
            topology_file: topology_input,
            system_file: system_input,
            integrator_file: integrator_input,
        }
        openmm_set = OpenMMSet(
            inputs=inputs,  # type: ignore
            topology_file=topology_file,
            system_file=system_file,
            integrator_file=integrator_file,
        )
        if Path(source_dir / state_file).is_file():
            openmm_set.inputs[state_file] = StateInput.from_file(source_dir / state_file)
            openmm_set.state_file = state_file  # should this be a dict-like assignment?
        return openmm_set

    def validate(self) -> bool:
        """
        Checks that the OpenMMSet will generate a valid simulation.

        Returns:
            bool
        """
        try:
            self.get_simulation()
        except Exception as e:
            print(
                "A valid simulation could not be generated, the following error was raised:",
                e,
            )
            return False
        else:
            return True

    def get_simulation(self) -> Simulation:
        """
        Instantiates and returns an OpenMM.Simulation from the input files.

        Returns:
            OpenMM.Simulation
        """
        topology_input = self.inputs[self.topology_file]
        system_input = self.inputs[self.system_file]
        integrator_input = self.inputs[self.integrator_file]
        simulation = Simulation(
            topology_input.get_topology(),  # type: ignore
            system_input.get_system(),  # type: ignore
            integrator_input.get_integrator(),  # type: ignore
        )
        if hasattr(self, "state_file") and self.state_file:
            # TODO: confirm that this works correctly
            state_input = self.inputs[self.state_file]
            simulation.context.setState(state_input.get_state())  # type: ignore
        return simulation


# noinspection PyMethodOverriding
class OpenMMGenerator(InputGenerator):
    """
    Generator for an OpenMM InputSet.

    This class is designed for generating simulations of mixed molecular systems. Starting with
    only SMILEs and counts, this class can generate a valid InputSet for a wide variety of
    molecular systems. Currently only the Sage force field is supported.

    This class is only compatible with the Langevin Middle Integrator. To use a different
    integrator, you can first generate the system with OpenMMGenerator and then add
    a different integrator to the OpenMMInputSet.
    """

    def __init__(
        self,
        force_field: str = "Sage",
        temperature: float = 298,
        step_size: float = 0.001,
        friction_coefficient: int = 1,
        partial_charge_scaling: Optional[Dict[str, float]] = None,
        partial_charges: Optional[List[Tuple[Union[pymatgen.core.Molecule, str, Path], np.ndarray]]] = None,
        packmol_random_seed: int = -1,
        topology_file: Union[str, Path] = "topology.pdb",
        system_file: Union[str, Path] = "system.xml",
        integrator_file: Union[str, Path] = "integrator.xml",
        state_file: Union[str, Path] = "state.xml",
    ):
        """
        Instantiates an OpenMMGenerator.

        Args:
            force_field: force field for parameterization, currently supported Force Fields: 'Sage'.
            temperature: the temperature to be added to the integrator (Kelvin).
            step_size: the step size of the simulation (picoseconds).
            friction_coefficient: the friction coefficient which couples the system to
                the heat bath (inverse picoseconds).
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
        self.partial_charge_scaling = partial_charge_scaling if partial_charge_scaling else {}
        self.partial_charges = partial_charges if partial_charges else []
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
        # create dynamic openmm objects with internal methods
        topology = self._get_openmm_topology(smiles)
        if box is None:
            box = self.get_box(smiles, density)  # type: ignore
        coordinates = self._get_coordinates(smiles, box, self.packmol_random_seed)
        smile_strings = list(smiles.keys())
        system = self._parameterize_system(
            topology,
            smile_strings,
            box,
            self.force_field,
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
    def _smile_to_molecule(smile: str) -> pymatgen.core.Molecule:
        """
        Convert a SMILE to a Pymatgen.Molecule.
        """
        mol = pybel.readstring("smi", smile)
        mol.addh()
        mol.make3D()
        adaptor = BabelMolAdaptor(mol.OBMol)
        return adaptor.pymatgen_mol

    @staticmethod
    def _smile_to_parmed_structure(smile: str) -> parmed.Structure:
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

    @staticmethod
    def _get_openmm_topology(smiles: Dict[str, int]) -> openmm.app.Topology:
        """
        Returns an openmm topology with the given SMILEs at the given counts.

        The topology does not contain coordinates.

        Parameters:
            smiles: keys are smiles and values are number of that molecule to pack

        Returns:
            an openmm.app.Topology
        """
        structures = [OpenMMGenerator._smile_to_parmed_structure(smile) for smile in smiles.keys()]
        counts = list(smiles.values())
        combined_structs = parmed.Structure()
        for struct, count in zip(structures, counts):
            combined_structs += struct * count
        return combined_structs.topology

    @staticmethod
    def _get_coordinates(smiles: Dict[str, int], box: List[float], random_seed: int) -> np.ndarray:
        """
        Pack the box with the molecules specified by smiles.

        Args:
            smiles: keys are smiles and values are number of that molecule to pack
            box: list of [xlo, ylo, zlo, xhi, yhi, zhi]

        Returns:
            array of coordinates for each atom in the box.
        """
        molecules = []
        for smile, count in smiles.items():
            molecules.append(
                {
                    "name": smile,
                    "number": count,
                    "coords": OpenMMGenerator._smile_to_molecule(smile),
                }
            )
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolBoxGen(seed=random_seed).get_input_set(molecules=molecules, box=box)
            pw.write_input(scratch_dir)
            pw.run(scratch_dir)
            coordinates = XYZ.from_file(pathlib.Path(scratch_dir, "packmol_out.xyz")).as_dataframe()
        raw_coordinates = coordinates.loc[:, "x":"z"].values  # type: ignore
        return raw_coordinates

    @staticmethod
    def get_box(smiles: Dict[str, int], density: float) -> List[float]:
        """
        Calculates the side_length of a cube necessary to contain the given molecules with
        given density.

        Args:
            smiles: keys are smiles and values are number of that molecule to pack
            density: guessed density of the solution, larger densities will lead to smaller cubes.

        Returns:
            side_length: side length of the returned cube
        """
        cm3_to_A3 = 1e24
        NA = 6.02214e23
        mols = [OpenMMGenerator._smile_to_molecule(smile) for smile in smiles.keys()]
        mol_mw = np.array([mol.composition.weight for mol in mols])
        counts = np.array(list(smiles.values()))
        total_weight = sum(mol_mw * counts)
        box_volume = total_weight * cm3_to_A3 / (NA * density)
        side_length = round(box_volume ** (1 / 3), 2)
        return [0, 0, 0, side_length, side_length, side_length]

    @staticmethod
    def _infer_openff_mol(charged_mol: Union[pymatgen.core.Molecule, str, Path]) -> openff.toolkit.topology.Molecule:
        if isinstance(charged_mol, (str, Path)):
            charged_mol = pymatgen.core.Molecule.from_file(str(charged_mol))
        with tempfile.NamedTemporaryFile() as f:
            # these next 4 lines are cursed
            pybel_mol = BabelMolAdaptor(charged_mol).pybel_mol  # pymatgen Molecule
            pybel_mol.write("mol2", filename=f.name, overwrite=True)  # pybel Molecule
            rdmol = rdkit.Chem.MolFromMol2File(f.name, removeHs=False)  # rdkit Molecule
        inferred_mol = openff.toolkit.topology.Molecule.from_rdkit(
            rdmol, hydrogens_are_explicit=True
        )  # OpenFF Molecule
        return inferred_mol

    @staticmethod
    def _get_atom_map(inferred_mol, openff_mol) -> Tuple[bool, Dict[int, int]]:
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

    @staticmethod
    def _add_mol_charges_to_forcefield(
        forcefield: smirnoff.ForceField,
        charged_openff_mol: List[openff.toolkit.topology.Molecule],
    ) -> smirnoff.ForceField:

        charge_type = LibraryChargeHandler.LibraryChargeType.from_molecule(charged_openff_mol)
        forcefield["LibraryCharges"].add_parameter(parameter=charge_type)
        return forcefield

    @staticmethod
    def _add_partial_charges_to_forcefield(
        forcefield: smirnoff.ForceField,
        smile_strings: List[str],
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
        for smile in smile_strings:
            # detect charge scaling, set scaling parameter
            if smile in partial_charge_scaling.keys():
                charge_scaling = partial_charge_scaling[smile]
            else:
                charge_scaling = 1
            # assign default am1bcc charges
            openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
            openff_mol.compute_partial_charges_am1bcc()
            openff_mol.partial_charges = openff_mol.partial_charges * charge_scaling
            # assign charges from isomorphic charges, if they exist
            is_isomorphic = False
            for mol_xyz, charges in partial_charges:
                inferred_mol = OpenMMGenerator._infer_openff_mol(mol_xyz)
                is_isomorphic, atom_map = OpenMMGenerator._get_atom_map(inferred_mol, openff_mol)
                # if is_isomorphic to a mol_xyz in the system, add to openff_mol else, warn user
                if is_isomorphic:
                    reordered_charges = np.array([charges[atom_map[i]] for i, _ in enumerate(charges)])
                    openff_mol.partial_charges = reordered_charges * charge_scaling * elementary_charge
                    break
            # return a warning if some partial charges were not matched to any mol_xyz
            if not is_isomorphic and len(partial_charges) > 0:
                warnings.warn(f"{mol_xyz} in partial_charges is not is_isomorphic to any SMILE in the system.")
            # finally, add charged mol to force_field
            OpenMMGenerator._add_mol_charges_to_forcefield(forcefield, openff_mol)
        return forcefield

    @staticmethod
    def _parameterize_system(
        topology: Topology,
        smile_strings: List[str],
        box: List[float],
        force_field: str,
        partial_charge_scaling: Dict[str, float],
        partial_charges: List[Tuple[Union[pymatgen.core.Molecule, str, Path], np.ndarray]],
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
        supported_force_fields = ["Sage"]
        if force_field.lower() == "sage":
            openff_mols = [openff.toolkit.topology.Molecule.from_smiles(smile) for smile in smile_strings]
            openff_forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
            openff_forcefield = OpenMMGenerator._add_partial_charges_to_forcefield(
                openff_forcefield,
                smile_strings,
                partial_charge_scaling,
                partial_charges,
            )
            openff_topology = openff.toolkit.topology.Topology.from_openmm(topology, openff_mols)
            box_vectors = list(np.array(box[3:6]) - np.array(box[0:3])) * angstrom
            openff_topology.box_vectors = box_vectors
            system = openff_forcefield.create_openmm_system(openff_topology)
            return system
        raise NotImplementedError(
            f"currently only these force fields are supported: {' '.join(supported_force_fields)}.\n"
            f"Please select one of the supported force fields."
        )
