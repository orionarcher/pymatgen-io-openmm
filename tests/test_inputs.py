# base python
import tempfile

# openmm
import openmm
from openmm.unit import femtoseconds, kelvin

# pymatgen
from pymatgen.io.openmm.inputs import (
    TopologyInput,
    SystemInput,
    IntegratorInput,
    StateInput,
)
from pymatgen.io.openmm.tests.datafiles import (
    topology_path,
    state_path,
    integrator_path,
    system_path,
)

__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


class TestInputFiles:
    def test_topology_input(self):
        topology_input = TopologyInput.from_file(topology_path)
        with tempfile.NamedTemporaryFile() as f:
            topology_input.write_file(f.name)
            topology_input = TopologyInput.from_file(f.name)
        topology = topology_input.get_topology()
        assert isinstance(topology, openmm.app.Topology)
        assert topology.getNumAtoms() == 780
        assert topology.getNumResidues() == 220
        assert topology.getNumBonds() == 560

    def test_system_input(self):
        system_input = SystemInput.from_file(system_path)
        with tempfile.NamedTemporaryFile() as f:
            system_input.write_file(f.name)
            system_input = SystemInput.from_file(f.name)
        system = system_input.get_system()
        assert isinstance(system, openmm.System)
        assert system.getNumParticles() == 780
        assert system.usesPeriodicBoundaryConditions()

    def test_integrator_input(self):
        integrator_input = IntegratorInput.from_file(integrator_path)
        with tempfile.NamedTemporaryFile() as f:
            integrator_input.write_file(f.name)
            integrator_input = IntegratorInput.from_file(f.name)
        integrator = integrator_input.get_integrator()
        assert isinstance(integrator, openmm.Integrator)
        assert integrator.getStepSize() == 1 * femtoseconds
        assert integrator.getTemperature() == 298 * kelvin

    def test_state_input(self):
        state_input = StateInput.from_file(state_path)
        with tempfile.NamedTemporaryFile() as f:
            state_input.write_file(f.name)
            state_input = StateInput.from_file(f.name)
        state = state_input.get_state()
        assert isinstance(state, openmm.State)
        assert len(state.getPositions(asNumpy=True)) == 780
