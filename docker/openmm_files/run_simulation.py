from openmm.app import Simulation
from openmm.app import PDBFile, StateDataReporter
from openmm.openmm import XmlSerializer
from sys import stdout

# load inputs
topology = PDBFile(open("topology.pdb").readlines()).getTopology()
system = XmlSerializer.deserialize(open("system.xml").read())
integrator = XmlSerializer.deserialize(open("integrator.xml").read())
state = XmlSerializer.deserialize(open("state.xml").read())

# setup simulation
simulation = Simulation(
    topology,
    system,
    integrator,
)
simulation.context.setState(state)
simulation.reporters.append(
    StateDataReporter(
        stdout,
        100,
        step=True,
        potentialEnergy=True,
        temperature=True,
        volume=True,
        density=True,
    )
)

# run simulation
simulation.minimizeEnergy()
simulation.step(10000)
