# base python
import tempfile

import monty
import monty.serialization

# cheminformatics
import numpy as np

from pymatgen.io.openmm.schema import InputMoleculeSpec

# openff

# openmm

# pymatgen

from pymatgen.io.openmm.sets import OpenMMSet
from pymatgen.io.openmm.generators import OpenMMSolutionGen, OpenMMAlchemyGen

# from pymatgen.io.openmm.tests.test_alchemy_utils import (
#     acetic_ethanol_hydrolysis,
#     acetic_ethanol_hydrolysis_del_water,
#     acetic_ethanol_select_dict,
# )
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
            "contents.json",
        }
        contents = input_set.inputs["contents.json"].contents
        assert set(contents.atom_resnames) == {"CCO", "H2O"}
        assert len(contents.atom_types) == 780
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
            default_charge_method="mmff94",
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
            "contents.json",
        }
        assert input_set.validate()

    def test_validation(self):
        pf6_charge_array = np.load(PF6_charges)
        li_charge_array = np.load(Li_charges)
        input_mol_dicts = [
            {"smile": "O", "count": 200, "name": "H2O"},
            {"smile": "CCO", "count": 20},
            {
                "smile": "[Li+]",
                "count": 10,
                "charge_scaling": 0.9,
                "geometries": [Li_xyz],
                "partial_charges": list(li_charge_array),
            },
            {
                "smile": "F[P-](F)(F)(F)(F)F",
                "count": 10,
                "charge_scaling": 0.9,
                "geometries": [PF6_xyz],
                "partial_charges": pf6_charge_array,
            },
        ]
        for mol_dict in input_mol_dicts:
            InputMoleculeSpec(**mol_dict)

    def test_get_input_set_w_charges(self):
        pf6_charge_array = np.load(PF6_charges)
        generator = OpenMMSolutionGen(packmol_random_seed=1, default_charge_method='mmff94')
        input_mol_dicts = [
            {"smile": "O", "count": 200, "name": "H2O"},
            {"smile": "CCO", "count": 20},
            {
                "smile": "[Li+]",
                "count": 10,
                "charge_scaling": 0.9,
                "forcefield": "Sage",
            },
            {
                "smile": "F[P-](F)(F)(F)(F)F",
                "count": 10,
                "charge_scaling": 0.9,
                "geometries": [PF6_xyz],
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
            "contents.json",
        }
        assert input_set.validate()


class TestOpenMMAlchemyGen:
    def test_get_alchemical_input_set(self, acetic_rxn):
        generator = OpenMMAlchemyGen(default_force_field="sage")
        input_mol_dicts = [
            {"smile": "O", "count": 20},
            {"smile": "CCO", "count": 40},
            {"smile": "CC(=O)O", "count": 40},
        ]
        # density is low to prevent a non bonded cutoff error
        input_set = generator.get_input_set(
            input_mol_dicts,
            reactions=[acetic_rxn],
            density=0.2,
        )
        assert input_set

    # TODO: add another reaction with multiple reactive sites
