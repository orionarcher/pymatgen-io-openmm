# base python

# cheminformatics
import numpy as np

# openff

# openmm

# pymatgen

from pymatgen.io.openmm.sets import OpenMMSet
from pymatgen.io.openmm.generators import OpenMMSolutionGen
from pymatgen.io.openmm.utils import get_openmm_topology
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

    def test_parameterize_system(self):
        # TODO: add test here to see if I am adding charges?
        topology = get_openmm_topology({"O": 200, "CCO": 20})
        smile_strings = ["O", "CCO"]
        box = [0, 0, 0, 19.59, 19.59, 19.59]
        force_field = "Sage"
        partial_charge_method = "am1bcc"
        system = OpenMMSolutionGen._parameterize_system(
            topology,
            smile_strings,
            box,
            force_field=force_field,
            partial_charge_method=partial_charge_method,
            partial_charge_scaling={},
            partial_charges=[],
        )
        assert system.getNumParticles() == 780
        assert system.usesPeriodicBoundaryConditions()

    def test_parameterize_mixedforcefield_system(self):
        # TODO: test with charges
        # TODO: periodic boundaries assertion
        # TODO: assert forcefields assigned correctly
        topology = get_openmm_topology({"O": 200, "CCO": 20})
        smile_strings = ["O", "CCO"]
        box = [0, 0, 0, 19.59, 19.59, 19.59]
        force_field = {"O": "amber14/spce.xml", "CCO": "gaff-2.11"}
        partial_charge_method = "am1bcc"
        system = OpenMMSolutionGen._parameterize_system(
            topology,
            smile_strings,
            box,
            force_field=force_field,
            partial_charge_method=partial_charge_method,
            partial_charge_scaling={},
            partial_charges=[],
        )
        assert system.getNumParticles() == 780
        assert system.usesPeriodicBoundaryConditions()

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
        big_smile = "O=C(OC(C)(C)CC/1=O)C1=C(O)/CCCCCCCC/C(NCCN(CCN)CCN)=C2C(OC(C)(C)CC/2=O)=O"
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
        input_set = generator.get_input_set({"O": 200, "CCO": 20, "F[P-](F)(F)(F)(F)F": 10, "[Li+]": 10}, density=1)
        assert isinstance(input_set, OpenMMSet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
        }
        assert input_set.validate()
