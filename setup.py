from openbabel import pybel
from openff.toolkit.typing.engines import smirnoff
from openmm.app import Simulation
from pymatgen.io.babel import BabelMolAdaptor

from pymatgen.io.packmol import PackmolBoxGen
from pymatgen.io.xyz import XYZ


import numpy as np
import parmed
import openff

import pathlib
import tempfile


def smile_to_mol(smile):
    """
    Converts a SMILE to a Pymatgen Molecule.

    Parameters
    ----------
    smile: a SMILE.

    Returns
    -------
    Pymatgen.Molecule
    """
    mol = pybel.readstring("smi", smile)
    mol.addh()
    mol.make3D()
    adaptor = BabelMolAdaptor(mol.OBMol)
    return adaptor.pymatgen_mol


class OpenMMSimulation:
    def __init__(self, *args, **kwargs):
        self.simulation = Simulation(*args, **kwargs)

    @staticmethod
    def _smile_to_parmed_structure(smile):
        """
        Converts a SMILE to a Parmed structure.

        Parameters
        ----------
        smile: a SMILE.

        Returns
        -------
        Parmed.Structure
        """
        mol = pybel.readstring("smi", smile)
        mol.addh()
        mol.make3D()
        with tempfile.NamedTemporaryFile() as f:
            mol.write(format="pdb", filename=f.name, overwrite=True)
            structure = parmed.load_file(f.name)
        return structure

    @staticmethod
    def _smiles_to_openmm_topology(smile_counts):
        """

        Parameters
        ----------
        smile_counts: a dictionary

        Returns
        -------

        """
        structure_counts = {
            count: OpenMMSimulation._smile_to_parmed_structure(smile)
            for count, smile in smile_counts.values()
        }
        combined_structs = parmed.Structure()
        for struct, count in structure_counts:
            combined_structs += struct * count
        return combined_structs.topology

    @staticmethod
    def _smiles_to_coordinates(smile_counts, box_size):
        molecules = []
        for smile, count in smile_counts.items():
            molecules.append({"name": smile, "number": count, "coords": smile_to_mol(smile)})
        with tempfile.TemporaryDirectory() as scratch_dir:
            pw = PackmolBoxGen().get_input_set(
                molecules=molecules
            )
            pw.write_input(scratch_dir)
            pw.run(scratch_dir)
            coordinates = XYZ.from_file(pathlib.Path(scratch_dir, "packmol_out.xyz")).as_dataframe()
        raw_coordinates = coordinates.loc[:, "x":"z"].values
        return raw_coordinates

    @staticmethod
    def _smiles_to_cube_size(smile_counts, density):
        """

        Parameters
        ----------
        smile_counts
        density

        Returns
        -------

        """
        cm3_to_A3 = 1e24
        NA = 6.02214e23
        mols = [smile_to_mol(smile) for smile in smile_counts.keys()]
        mol_mw = np.array([mol.structure.composition.weight for mol in mols])
        counts = np.array(smile_counts.values())
        total_weight = sum(mol_mw * counts)
        box_volume = total_weight * cm3_to_A3 / (NA * density)
        side_length = box_volume ** (1 / 3)
        return side_length

    @staticmethod
    def _smiles_to_system(smile_counts, density=1.5):
        topology = OpenMMSimulation._smiles_to_openmm_topology(smile_counts)
        box_size = OpenMMSimulation._smiles_to_cube_size(smile_counts, density)
        openff_mols = [
            openff.toolkit.topology.Molecule.from_smiles(smile) for smile in smiles
        ]
        openff_forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
        openff_topology = openff.toolkit.topology.Topology.from_openmm(
            topology, openff_mols
        )
        openff_topology.box_vectors = [box_size, box_size, box_size] * angstrom
        system = openff_forcefield.create_openmm_system(openff_topology)
        return system, topology


    def from_smiles