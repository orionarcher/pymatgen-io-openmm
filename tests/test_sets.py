# base python
import tempfile

# pymatgen
from pymatgen.io.openmm.inputs import (
    TopologyInput,
    StateInput,
)
from pymatgen.io.openmm.sets import OpenMMSet

from pymatgen.io.openmm.tests.datafiles import (
    input_set_path,
    corrupted_state_path,
)

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


class TestOpenMMSet:
    def test_from_directory(self):
        input_set = OpenMMSet.from_directory(input_set_path)
        assert len(input_set.inputs) == 4
        assert input_set.topology_file == "topology.pdb"
        assert input_set.state_file == "state.xml"
        assert isinstance(input_set.inputs["topology.pdb"], TopologyInput)
        assert isinstance(input_set.inputs["state.xml"], StateInput)
        input_set2 = OpenMMSet.from_directory(input_set_path, state_file="wrong_file.xml")
        assert len(input_set2.inputs) == 3
        assert input_set2.topology_file == "topology.pdb"
        assert input_set2.state_file is None

    def test_validate(self):
        input_set = OpenMMSet.from_directory(input_set_path)
        assert input_set.validate()
        corrupted_input_set = OpenMMSet.from_directory(input_set_path, state_file=corrupted_state_path)
        assert not corrupted_input_set.validate()

    def test_get_simulation(self):
        input_set = OpenMMSet.from_directory(input_set_path)
        simulation = input_set.get_simulation()
        state = simulation.context.getState(getPositions=True)
        assert len(state.getPositions(asNumpy=True)) == 780
        assert simulation.system.getNumParticles() == 780
        assert simulation.system.usesPeriodicBoundaryConditions()
        assert simulation.topology.getNumAtoms() == 780
        assert simulation.topology.getNumResidues() == 220
        assert simulation.topology.getNumBonds() == 560

    def test_write_inputs(self):
        input_set = OpenMMSet.from_directory(input_set_path)
        with tempfile.TemporaryDirectory() as scratch_dir:
            input_set.write_input(scratch_dir)
            input_set2 = OpenMMSet.from_directory(scratch_dir)
        assert len(input_set2.inputs) == 4
        assert input_set2.topology_file == "topology.pdb"
        assert input_set2.state_file == "state.xml"
        assert isinstance(input_set2.inputs["topology.pdb"], TopologyInput)
        assert isinstance(input_set2.inputs["state.xml"], StateInput)
