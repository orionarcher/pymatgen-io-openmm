"""
This functionality eventually belongs in Atomate, but for the sake of rapidly developing
a OpenMM IO, I will leave it here for now.

The idea is that each function will operate on a openmm.Simulation and propagate the
simulation forward in time.
"""

# base python
from typing import Union, Tuple, Optional, Dict

# scipy
import numpy as np

# openmm
from openmm.openmm import MonteCarloBarostat, Platform
from openmm.unit import atmosphere, kelvin
from openmm.app import Simulation


__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"

from pymatgen.io.openmm.sets import OpenMMAlchemySet
from pymatgen.io.openmm.utils import parameterize_w_interchange
from pymatgen.io.openmm.inputs import (
    TopologyInput,
    SystemInput,
    IntegratorInput,
    StateInput,
    ReactiveSystemInput,
)


def react_system(
    input_set: OpenMMAlchemySet,
    n_cycles: int = 1,
    steps_per_cycle: int = 1000,
    initial_steps: int = 0,
    cutoff_distance: float = 4,
    platform: Optional[Union[str, Platform]] = None,
    platformProperties: Optional[Dict[str, str]] = None,
):
    simulation = input_set.get_simulation(
        platform=platform, platformProperties=platformProperties
    )
    openmm_topology = None
    openmm_system = None

    # initial minimization and equilibration
    simulation.minimizeEnergy()
    simulation.step(initial_steps)

    reactive_system = input_set.inputs[input_set.reactive_system_file].reactive_system

    for cycle in range(n_cycles):
        # get positions
        state = simulation.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True)._value * 10

        # react system, generate mapping
        old_to_new_1 = reactive_system.react(positions, cutoff_distance=cutoff_distance)
        openff_topology, old_to_new_2 = reactive_system.generate_topology(
            update_self=True, return_index=True
        )

        # use mapping to update positions, dicts must be sorted by key
        old_to_new_arr_1 = np.array([i for i in old_to_new_1.keys()])
        old_to_new_arr_2 = np.array([i for i in old_to_new_2.keys()])
        new_positions = positions[old_to_new_arr_1][old_to_new_arr_2]
        # TODO: confirm this works

        # charges must be assigned with mmff94
        mol_specs = [{"openff_mol": mol} for mol in openff_topology.unique_molecules]
        for spec in mol_specs:
            spec["openff_mol"].assign_partial_charges("mmff94")

        box_matrix = state.getPeriodicBoxVectors(asNumpy=True)._value
        box = np.concatenate([[0, 0, 0], np.diag(box_matrix)]) * 10
        openmm_system = parameterize_w_interchange(openff_topology, mol_specs, box)
        openmm_topology = openff_topology.to_openmm()

        # create and evolve simulation
        simulation = Simulation(
            openmm_topology,
            openmm_system,
            input_set[input_set.integrator_file].get_integrator(),
            platform=simulation.context.getPlatform(),
            platformProperties=platformProperties,
        )
        simulation.context.setPositions(np.divide(new_positions, 10))
        simulation.minimizeEnergy()
        simulation.step(steps_per_cycle)

    state = simulation.context.getState(getPositions=True, getVelocities=True)
    # instantiate input files and feed to input_set
    input_set = OpenMMAlchemySet(
        inputs={
            input_set.topology_file: TopologyInput(openmm_topology),
            input_set.system_file: SystemInput(openmm_system),
            input_set.integrator_file: IntegratorInput(simulation.integrator),
            input_set.state_file: StateInput(state),
            input_set.reactive_system_file: ReactiveSystemInput(reactive_system),
        },
        topology_file=input_set.topology_file,
        system_file=input_set.system_file,
        integrator_file=input_set.integrator_file,
        state_file=input_set.state_file,
    )
    return input_set


def equilibrate_pressure(
    simulation: Simulation,
    steps: int,
    temperature: float = 298,
    pressure: float = 1,
    frequency: int = 10,
):
    """
    Equilibrate the pressure of a simulation in the NPT ensemble.

    Adds and then removes an openmm.MonteCarlo Barostat to shift the system
    into the NPT ensemble.

    Parameters
    ----------
    simulation : the openmm.Simulation to propagate.
    steps : the length of the heating, holding, and cooling stages. Steps is number of steps.
    temperature : temperature to equilibrate at (Kelvin)
    pressure : pressure to equilibrate at (Atm).
    frequency : the frequency at which pressure changes should be attempted (steps).
    """
    context = simulation.context
    system = context.getSystem()
    assert (
        system.usesPeriodicBoundaryConditions()
    ), "system must use periodic boundary conditions for pressure equilibration."
    integrator = simulation.context.getIntegrator()
    integrator.setTemperature(temperature * kelvin)
    barostat_force_index = system.addForce(
        MonteCarloBarostat(pressure * atmosphere, temperature * kelvin, frequency)
    )
    context.reinitialize(preserveState=True)
    simulation.step(steps)
    system.removeForce(barostat_force_index)
    context.reinitialize(preserveState=True)


def anneal(
    simulation: Simulation,
    temperature: Union[float, int],
    steps: Tuple[int, int, int],
    temp_steps: int = 100,
):
    """
    Anneal the simulation at the specified temperature.

    Annealing takes place in 3 stages, heating, holding, and cooling. The three
    elements of steps specify the length of each stage. After heating, and holding,
    the system will cool to its original temperature.

    Parameters
    ----------
    simulation : the openmm.Simulation to propagate.
    temperature : the temperature reached for the holding stage (Kelvin)
    steps : the length of the heating, holding, and cooling stages. Steps is number of steps.
    temp_steps : temperature is raised and lowered stepwise in temp_steps pieces.
    """
    # TODO: timing is currently bugged and not propagating long enough, should be fixed
    assert len(steps) == 3, ""
    integrator = simulation.context.getIntegrator()
    old_temperature = integrator.getTemperature()
    temp_step_size = abs(temperature * kelvin - old_temperature) / temp_steps

    for temp in np.arange(
        old_temperature + temp_step_size,
        temperature * kelvin + temp_step_size,
        temp_step_size,
    ):
        integrator.setTemperature(temp * kelvin)
        simulation.step(steps[0] // temp_steps)

    simulation.step(steps[1])

    for temp in np.arange(
        temperature * kelvin - temp_step_size,
        old_temperature - temp_step_size,
        -1 * temp_step_size,
    ):
        integrator.setTemperature(temp * kelvin)
        simulation.step(steps[2] // temp_steps)
