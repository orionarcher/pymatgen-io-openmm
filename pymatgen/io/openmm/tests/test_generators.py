# base python

# cheminformatics

import numpy as np


# openff

# openmm

# pymatgen

from pymatgen.io.openmm.sets import OpenMMSet
from pymatgen.io.openmm.generators import OpenMMSolutionGen, OpenMMAlchemyGen

# from pymatgen.io.openmm.tests.test_alchemy_utils import (
#     acetic_ethanol_hydrolysis,
#     acetic_ethanol_hydrolysis_del_water,
#     select_dict,
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
        input_set = generator.get_input_set({"O": 200, "CCO": 20}, density=1)
        assert isinstance(input_set, OpenMMSet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
        }
        assert input_set.validate()

    def test_get_input_set_big_smile(self):
        generator = OpenMMSolutionGen(
            packmol_random_seed=1,
            partial_charge_method="mmff94",
        )
        big_smile = (
            "O=C(OC(C)(C)CC/1=O)C1=C(O)/CCCCCCCC/C(NCCN(CCN)CCN)=C2C(OC(C)(C)CC/2=O)=O"
        )
        input_set = generator.get_input_set({"O": 200, big_smile: 1}, density=1)
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
        generator = OpenMMSolutionGen(
            partial_charges=[(PF6_xyz, pf6_charge_array), (Li_xyz, li_charge_array)],
            partial_charge_scaling={"Li": 0.9, "PF6": 0.9},
            packmol_random_seed=1,
        )
        input_set = generator.get_input_set(
            {"O": 200, "CCO": 20, "F[P-](F)(F)(F)(F)F": 10, "[Li+]": 10}, density=1
        )
        assert isinstance(input_set, OpenMMSet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
        }
        assert input_set.validate()

    def test_get_input_set_w_charges_and_forcefields(self):
        pf6_charge_array = np.load(PF6_charges)
        li_charge_array = np.load(Li_charges)
        generator = OpenMMSolutionGen(
            partial_charges=[(PF6_xyz, pf6_charge_array), (Li_xyz, li_charge_array)],
            partial_charge_scaling={"Li": 0.9, "PF6": 0.9},
            packmol_random_seed=1,
            force_field={"O": "spce"},
        )
        input_set = generator.get_input_set(
            {"O": 200, "CCO": 20, "F[P-](F)(F)(F)(F)F": 10, "[Li+]": 10}, density=1
        )
        assert isinstance(input_set, OpenMMSet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
        }
        assert input_set.validate()

    def test_formal_charge(self):

        trimer_smile = (
            "O=C1[C@H]([C@H](OC(O)=C1/C(CCCCCCCC/C(O[H])=C2C(["
            "C@H]([C@H](OC/2=O)C(C)C)C)=O)=[NH+]/CCN(CC/[NH+]=C("
            r"C3=C(O[C@@H]([C@@H](C3=O)C)C(C)C)O)\CCCCCCCC/C(O["
            "H])=C4C([C@H]([C@H](OC/4=O)C(C)C)C)=O)CC/[NH+]=C("
            r"C5=C(O[C@@H]([C@@H](C5=O)C)C(C)C)O)\CCCCCCCC/C(O["
            "H])=C6C([C@H]([C@H](OC/6=O)C(C)C)C)=O)C(C)C)C "
        )
        openmm_generator = OpenMMSolutionGen(
            temperature=298,
            step_size=0.001,
            partial_charge_method="mmff94",
            force_field={"O": "spce", trimer_smile: "sage"},
        )

        molecules = {"O": 200, trimer_smile: 1}

        input_set = openmm_generator.get_input_set(molecules, density=0.5)

        assert isinstance(input_set, OpenMMSet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
        }
        assert input_set.validate()


class TestOpenMMAlchemyGen:
    def test_get_alchemical_input_set(self, acetic_ethanol_condensation):
        generator = OpenMMAlchemyGen(force_field="sage")
        input_set = generator.get_input_set(
            {"O": 200, "CC(=O)O": 10, "CCO": 10},
            reaction=acetic_ethanol_condensation,
            density=1,
        )
        assert input_set

    # TODO: add another reaction with multiple reactive sites
