from pkg_resources import resource_filename
import pathlib
from pathlib import Path

topology_path = resource_filename(__name__, "test_files/input_set_files/topology.pdb")
system_path = resource_filename(__name__, "test_files/input_set_files/system.xml")
integrator_path = resource_filename(__name__, "test_files/input_set_files/integrator.xml")
state_path = resource_filename(__name__, "test_files/input_set_files/state.xml")

coordinates_path = resource_filename(__name__, "test_files/water_ethanol_coordinates.npy")
