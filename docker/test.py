import pathlib
import os

from pymatgen.io.openmm.sets import OpenMMSet
from openmm import Platform

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

sim.minimizeEnergy()

print("Minimized energy")

sim.step(10000)

print("stepped 10000")
