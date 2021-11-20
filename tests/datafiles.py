from pkg_resources import resource_filename
import pathlib
from pathlib import Path

topology_path = resource_filename(__name__, "test_files/input_set/topology.pdb")
system_path = resource_filename(__name__, "test_files/input_set/system.xml")
integrator_path = resource_filename(__name__, "test_files/input_set/integrator.xml")
state_path = resource_filename(__name__, "test_files/input_set/state.xml")

coordinates_path = resource_filename(__name__, "test_files/water_ethanol_coordinates.npy")
