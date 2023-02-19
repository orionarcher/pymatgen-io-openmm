# base python
import tempfile

import monty
import monty.serialization

# cheminformatics
import numpy as np

# openff

# openmm

# pymatgen

from pymatgen.io.openmm.sets import OpenMMSet
from pymatgen.io.openmm.generators import OpenMMSolutionGen
from pymatgen.io.openmm.tests.datafiles import (
    PF6_xyz,
    PF6_charges,
    Li_charges,
    Li_xyz,
)

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


class TestOpenMMSolutionGen:
    #  TODO: add test for formally charged smile

    def test_get_input_set(self):
        generator = OpenMMSolutionGen(packmol_random_seed=1)
        # TODO: figure out why tests are failing if density is 1
        input_mol_dicts = [
            {"smile": "O", "count": 200, "name": "H2O"},
            {"smile": "CCO", "count": 20},
        ]
        input_set = generator.get_input_set(input_mol_dicts, density=0.8)
        assert isinstance(input_set, OpenMMSet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
        }
        assert set(input_set.settings["atom_resnames"]) == {"CCO", "H2O"}
        assert len(input_set.settings["atom_types"]) == 780
        monty.serialization.dumpfn(input_set, "./test")
        assert input_set.validate()

    def test_dump_load_input_set(self):
        generator1 = OpenMMSolutionGen(packmol_random_seed=1)
        with tempfile.TemporaryDirectory() as tmpdir:

            monty.serialization.dumpfn(generator1, tmpdir + "/generator.json")
            generator2 = monty.serialization.loadfn(tmpdir + "/generator.json")

        assert generator1.as_dict() == generator2.as_dict()

    def test_get_input_set_big_smile(self):
        generator = OpenMMSolutionGen(
            packmol_random_seed=1,
            partial_charge_method="mmff94",
        )
        big_smile = (
            "O=C(OC(C)(C)CC/1=O)C1=C(O)/CCCCCCCC/C(NCCN(CCN)CCN)=C2C(OC(C)(C)CC/2=O)=O"
        )
        input_mol_dicts = [
            {"smile": "O", "count": 200},
            {"smile": big_smile, "count": 1},
        ]
        input_set = generator.get_input_set(input_mol_dicts, density=0.8)
        assert isinstance(input_set, OpenMMSet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
        }
        assert input_set.validate()

    def test_get_input_set_w_charges(self):
        pf6_charge_array = np.load(PF6_charges)
        li_charge_array = np.load(Li_charges)
        generator = OpenMMSolutionGen(packmol_random_seed=1)
        input_mol_dicts = [
            {"smile": "O", "count": 200, "name": "H2O"},
            {"smile": "CCO", "count": 20},
            {
                "smile": "[Li+]",
                "count": 10,
                "charge_scaling": 0.9,
                "geometries": Li_xyz,
                "partial_charges": li_charge_array,
            },
            {
                "smile": "F[P-](F)(F)(F)(F)F",
                "count": 10,
                "charge_scaling": 0.9,
                "geometries": PF6_xyz,
                "partial_charges": pf6_charge_array,
            },
        ]
        input_set = generator.get_input_set(input_mol_dicts, density=1)
        assert isinstance(input_set, OpenMMSet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
        }
        assert input_set.validate()

    def test_get_input_set_w_geometries(self):
        # TODO: add test for initial_geometries kwarg
        return
