"""
This functionality eventually belongs in Atomate, but for the sake of rapidly developing
a OpenMM IO, I will leave it here for now.

The idea is that each function will operate on a openmm.Simulation and propagate the
simulation forward in time.
"""

from typing import Union, List

from openmm import MonteCarloBarostat
from openmm.unit import kelvin, atmosphere
from openmm.app import Simulation

import numpy as np


def equilibrate_pressure(
    simulation: Simulation,
    steps: int,
):
    """
    Equilibrate the pressure of a simulation in the NPT ensemble.

    Adds and then removes a openmm.MonteCarloBarostat to shift the system
    into the NPT ensemble.

    Parameters
    ----------
    simulation : the openmm.Simulation to propagate.
    steps : the length of the heating, holding, and cooling stages.
    """
    context = simulation.context
    system = context.getSystem()
    assert (
        system.usesPeriodicBoundaryConditions()
    ), "system must use periodic boundary conditions for pressure equilibration."
    barostat_force_index = system.addForce(MonteCarloBarostat(1 * atmosphere, 298 * kelvin, 10))
    context.reinitialize(preserveState=True)
    simulation.step(steps)
    system.removeForce(barostat_force_index)
    context.reinitialize(preserveState=True)


def anneal(
    simulation: Simulation,
    temperature: Union[float, int],
    steps: List[int],
    temp_steps: int = 100,
):
    """
    Anneal the simulation at the specified temperature.

    Annealing takes place in 3 stages, heating, holding, and cooling. The three
    elements of steps specify the length of each stage.

    Parameters
    ----------
    simulation : the openmm.Simulation to propagate.
    temperature : the temperature reached for the holding stage.
    steps : the length of the heating, holding, and cooling stages.
    temp_steps : temperature is raised and lowered stepwise in temp_steps pieces.
    """
    # TODO: timing is currently bugged and not propagating long enough, should be fixed
    assert len(steps) == 3, ""
    integrator = simulation.context.getIntegrator()
    old_temperature = integrator.getTemperature()
    temp_step_size = abs(temperature * kelvin - old_temperature) / temp_steps

    for temp in np.arange(old_temperature, temperature * kelvin + temp_step_size, temp_step_size):
        integrator.setTemperature(temp * kelvin)
        simulation.step(steps[0] // temp_steps)

    simulation.step(steps[1])

    for temp in np.arange(temperature * kelvin, old_temperature - temp_step_size, temp_step_size):
        integrator.setTemperature(temp * kelvin)
        simulation.step(steps[2] // temp_steps)
