"""
This functionality eventually belongs in Atomate, but for the sake of rapidly developing
a OpenMM IO, I will leave it here for now.

The idea is that each function will operate on a openmm.Simulation and propagate the
simulation forward in time.
"""

# base python
from typing import Union, List

# scipy
import numpy as np

# openmm
from openmm import MonteCarloBarostat
from openmm.unit import kelvin, atmosphere
from openmm.app import Simulation


__author__ = "Orion Cohen, Ryan Kingsbury"
__version__ = "1.0"
__maintainer__ = "Orion Cohen"
__email__ = "orion@lbl.gov"
__date__ = "Nov 2021"


def equilibrate_pressure(
    simulation: Simulation,
    steps: int,
    temperature: float,
    pressure: float,
):
    """
    Equilibrate the pressure of a simulation in the NPT ensemble.

    Adds and then removes a openmm.MonteCarlo Barostat to shift the system
    into the NPT ensemble.

    Parameters
    ----------
    simulation : the openmm.Simulation to propagate.
    steps : the length of the heating, holding, and cooling stages. Steps is number of steps.
    temperature : temperature to equilibrate at (Kelvin)
    pressure : pressure to equilibrate at (Atm).
    """
    context = simulation.context
    system = context.getSystem()
    assert (
        system.usesPeriodicBoundaryConditions()
    ), "system must use periodic boundary conditions for pressure equilibration."
    barostat_force_index = system.addForce(MonteCarloBarostat(pressure * atmosphere, temperature * kelvin, 10))
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
    temperature : the temperature reached for the holding stage (Kelvin)
    steps : the length of the heating, holding, and cooling stages. Steps is number of steps.
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
