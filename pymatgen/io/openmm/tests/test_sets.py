# base python
import tempfile

import monty
import monty.serialization

# pymatgen
from pymatgen.io.openmm.inputs import (
    TopologyInput,
    StateInput,
)
from pymatgen.io.openmm.sets import OpenMMSet

from pymatgen.io.openmm.tests.datafiles import (
    input_set_dir,
    corrupted_state_path,
    input_set_path,
)

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


class TestOpenMMSet:
    def test_from_directory(self):
        input_set = OpenMMSet.from_directory(input_set_dir)
        assert len(input_set.inputs) == 4
        assert input_set.topology_file == "topology.pdb"
        assert input_set.state_file == "state.xml"
        assert isinstance(input_set.inputs["topology.pdb"], TopologyInput)
        assert isinstance(input_set.inputs["state.xml"], StateInput)
        input_set2 = OpenMMSet.from_directory(input_set_dir, state_file="wrong_file.xml")
        assert len(input_set2.inputs) == 3
        assert input_set2.topology_file == "topology.pdb"
        assert input_set2.get("state_file") is None

    def test_dump_load_input_set(self):

        input_set1 = OpenMMSet.from_directory(input_set_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            monty.serialization.dumpfn(input_set1, tmpdir + "/input_set.json")
            input_set2 = monty.serialization.loadfn(tmpdir + "/input_set.json")

        assert input_set1.keys() == input_set2.keys()

        topology1 = input_set1.inputs["topology.pdb"].topology
        topology2 = input_set2.inputs["topology.pdb"].topology
        assert topology1 == topology2

        for file in ["state.xml", "integrator.xml", "system.xml"]:
            openmm_object1 = input_set1.inputs[file].openmm_object
            openmm_object2 = input_set2.inputs[file].openmm_object
            assert openmm_object1 == openmm_object2

    def test_validate(self):
        input_set = OpenMMSet.from_directory(input_set_dir)
        assert input_set.validate()
        corrupted_input_set = OpenMMSet.from_directory(input_set_dir, state_file=corrupted_state_path)
        assert not corrupted_input_set.validate()

    def test_get_simulation(self):
        input_set = OpenMMSet.from_directory(input_set_dir)
        simulation = input_set.get_simulation()
        state = simulation.context.getState(getPositions=True)
        assert len(state.getPositions(asNumpy=True)) == 780
        assert simulation.system.getNumParticles() == 780
        assert simulation.system.usesPeriodicBoundaryConditions()
        assert simulation.topology.getNumAtoms() == 780
        assert simulation.topology.getNumResidues() == 220
        assert simulation.topology.getNumBonds() == 560

    def test_write_inputs(self):
        input_set = OpenMMSet.from_directory(input_set_dir)
        with tempfile.TemporaryDirectory() as scratch_dir:
            input_set.write_input(scratch_dir)
            input_set2 = OpenMMSet.from_directory(scratch_dir)
        assert len(input_set2.inputs) == 4
        assert input_set2.topology_file == "topology.pdb"
        assert input_set2.state_file == "state.xml"
        assert isinstance(input_set2.inputs["topology.pdb"], TopologyInput)
        assert isinstance(input_set2.inputs["state.xml"], StateInput)
