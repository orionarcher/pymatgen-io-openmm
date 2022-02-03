# base python
import json
import tempfile

# pymatgen
from monty.json import MontyDecoder

from pymatgen.io.openmm.inputs import (
    TopologyInput,
    StateInput,
)
from pymatgen.io.openmm.sets import OpenMMSet, OpenMMAlchemySet

from pymatgen.io.openmm.tests.datafiles import (
    input_set_dir,
    corrupted_state_path,
    alchemy_input_set_path,
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
        input_set2 = OpenMMSet.from_directory(
            input_set_dir, state_file="wrong_file.xml"
        )
        assert len(input_set2.inputs) == 3
        assert input_set2.topology_file == "topology.pdb"
        assert input_set2.state_file is None

    def test_validate(self):
        input_set = OpenMMSet.from_directory(input_set_dir)
        assert input_set.validate()
        corrupted_input_set = OpenMMSet.from_directory(
            input_set_dir, state_file=corrupted_state_path
        )
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


class TestOpenMMAlchemySet:
    def test_from_directory(self):
        """
        to do
        """
        input_set = OpenMMAlchemySet.from_directory(alchemy_input_set_path)
        assert isinstance(input_set, OpenMMAlchemySet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
            "reaction_spec.json",
        }
        assert input_set.validate()
        rxn_spec = json.loads(input_set.inputs["reaction_spec.json"], cls=MontyDecoder)
        assert len(rxn_spec["trigger_atoms"][0]) == 10
        assert len(rxn_spec["trigger_atoms"][1]) == 10
        assert len(rxn_spec["half_reactions"]) == 20
        assert rxn_spec["force_field"] == "sage"
