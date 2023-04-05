import time
import pathlib
import os

from openmm.app import DCDReporter, StateDataReporter, PDBReporter
from pymatgen.io.openmm.sets import OpenMMSet
from openmm import Platform
from pymatgen.io.openmm.simulations import anneal, equilibrate_pressure
from pymatgen.io.openmm.inputs import StateInput

wd = pathlib.Path.cwd()
input_dir = pathlib.Path("/input")
output_dir = pathlib.Path("/output")

rank = int(os.environ.get("SLURM_PROCID"))  # type: ignore

print("SLURM_PROCID: ", rank)

target_paths = [path for path in input_dir.iterdir()]
assert len(target_paths) == 4, "there must be exactly 4 jobs to run"

target_path = target_paths[rank]

print("Target path is: ", target_path)

input_set = OpenMMSet.from_directory(target_path)

print("Created input_set")

platform = Platform.getPlatformByName("CUDA")
sim = input_set.get_simulation(
    platform=platform,
    platformProperties={"DeviceIndex": str(rank)},
)

print("Created simulation")

output_dir = output_dir / target_path.stem
output_dir.mkdir(parents=True, exist_ok=True)
pdb_reporter = PDBReporter(str(output_dir / "topology.pdb"), 1)
pdb_reporter.report(sim, sim.context.getState(getPositions=True))
sim.reporters.append(
    StateDataReporter(
        str(output_dir / "state.txt"),
        1000,
        step=True,
        potentialEnergy=True,
        temperature=True,
        volume=True,
        density=True,
    )
)
sim.reporters.append(DCDReporter(str(output_dir / "trajectory_equil.dcd"), 10000))

start = time.time()
sim.minimizeEnergy()
equilibrate_pressure(sim, 1000000)
anneal(sim, 400, [1000000, 1000000, 1000000])

sim.reporters.pop()
sim.reporters.append(DCDReporter(str(output_dir / "trajectory.dcd"), 10000))

sim.step(5000000)

# write out final output set
state = sim.context.getState(getPositions=True, getVelocities=True)
input_set.inputs["state.xml"] = StateInput(state)
input_set.write_input(output_dir / f"output_set_{target_path.stem}")

print("total runtime: ", time.time() - start)
