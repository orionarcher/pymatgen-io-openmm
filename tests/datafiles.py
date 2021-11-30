"""
Exists to organize testing data by providing standard links to each resource.
"""

from pathlib import Path
from pkg_resources import resource_filename  # type: ignore

tests_path = Path(__file__).parent
test_files_path = Path("test_files")
input_set_path = test_files_path / "input_set"
partial_charges = test_files_path / "partial_charge"


topology_path = resource_filename(__name__, str(input_set_path / "topology.pdb"))
system_path = resource_filename(__name__, str(input_set_path / "system.xml"))
integrator_path = resource_filename(__name__, str(input_set_path / "integrator.xml"))
state_path = resource_filename(__name__, str(input_set_path / "state.xml"))
corrupted_state_path = resource_filename(__name__, str(input_set_path / "corrupted_state.xml"))
input_set_dir = resource_filename(__name__, str(input_set_path))

coordinates_path = resource_filename(__name__, str(test_files_path / "water_ethanol_coordinates.npy"))

CCO_xyz = resource_filename(__name__, str(partial_charges / "CCO.xyz"))
CCO_charges = resource_filename(__name__, str(partial_charges / "CCO.npy"))
FEC_r_xyz = resource_filename(__name__, str(partial_charges / "FEC-r.xyz"))
FEC_s_xyz = resource_filename(__name__, str(partial_charges / "FEC-s.xyz"))
FEC_charges = resource_filename(__name__, str(partial_charges / "FEC.npy"))
PF6_xyz = resource_filename(__name__, str(partial_charges / "PF6.xyz"))
PF6_charges = resource_filename(__name__, str(partial_charges / "PF6.npy"))
