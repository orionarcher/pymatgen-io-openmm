import unittest
import openff.toolkit as tk
from pymatgen.io.openmm.parameterizer import Parameterizer, ParameterizerType, ParameterizerAssignment
from typing import List, Optional, Dict


class TestParameterizer(unittest.TestCase):
    def setUp(self):
        self.topology = tk.Topology.from_molecules([tk.Molecule.from_smiles("CCCC")])
        self.mol_specs = [{"openff_mol": tk.Molecule.from_smiles("CCCC"),
                          "charge_group_idxs": [[0, 1], [2, 3]], 
                          "vdw_group_idxs": [[0, 1], [2, 3]], 
                          "rotatable_bond_idxs": [[1, 2]]}]
        self.box = [0, 0, 0, 10, 10, 10]
        self.force_fields = ["sage"]
        self.parameterizer_assignment = ParameterizerAssignment.INFERRED
        self.parameterizer_type = ParameterizerType.OPENMM_PARAMETERIZER
        self.customize_force_field = False
        self.custom_file_paths = None

    def test_parameterizer_initialization(self):
        parameterizer = Parameterizer(self.topology, self.mol_specs, self.box, self.force_fields,
                                      parameterizer_type=self.parameterizer_type,
                                      parameterizer_assignment=self.parameterizer_assignment,
                                      customize_force_field=self.customize_force_field,
                                      custom_file_paths=self.custom_file_paths)

        self.assertEqual(parameterizer.parameterizer_type, ParameterizerType.INTERCHANGE_PARAMETERIZER)
        self.assertEqual(parameterizer.force_fields, self.force_fields)
        self.assertEqual(parameterizer.mol_specs, self.mol_specs)
        self.assertEqual(parameterizer.box, self.box)
        self.assertEqual(parameterizer.using_custom_files, False)
        self.assertEqual(parameterizer.custom_force_field_files, None)
        self.assertEqual(parameterizer.additional_args, {})
       

    def test_custom_file_paths_with_customize_force_field(self):
        self.customize_force_field = True
        self.custom_file_paths = ["custom.xml"]
        with self.assertRaises(AssertionError):
            Parameterizer(self.topology, self.mol_specs, self.box, self.force_fields,
                          parameterizer_type=self.parameterizer_type,
                          parameterizer_assignment=self.parameterizer_assignment,
                          customize_force_field=self.customize_force_field,
                          custom_file_paths=self.custom_file_paths)

    def test_no_custom_file_paths_with_customize_force_field(self):
        self.customize_force_field = True
        self.custom_file_paths = None
        with self.assertRaises(AssertionError):
            Parameterizer(self.topology, self.mol_specs, self.box, self.force_fields,
                          parameterizer_type=self.parameterizer_type,
                          parameterizer_assignment=self.parameterizer_assignment,
                          customize_force_field=self.customize_force_field,
                          custom_file_paths=self.custom_file_paths)

    def test_explicit_parameterizer_type_not_provided(self):
        self.parameterizer_assignment = ParameterizerAssignment.EXPLICIT
        with self.assertRaises(AssertionError):
            Parameterizer(self.topology, self.mol_specs, self.box, self.force_fields,
                          parameterizer_type=None,
                          parameterizer_assignment=self.parameterizer_assignment,
                          customize_force_field=self.customize_force_field,
                          custom_file_paths=self.custom_file_paths)

    def test_force_field_type_not_supported_by_parameterizer(self):
        self.parameterizer_type = ParameterizerType.OPENMM_PARAMETERIZER
        with self.assertRaises(AssertionError):
            Parameterizer(self.topology, self.mol_specs, self.box, self.force_fields,
                          parameterizer_type=self.parameterizer_type,
                          parameterizer_assignment=ParameterizerAssignment.EXPLICIT,
                          customize_force_field=self.customize_force_field,
                          custom_file_paths=self.custom_file_paths)

    def test_unexpected_topology_type_for_parameterizer_type(self):
        self.parameterizer_type = ParameterizerType.OPENMM_PARAMETERIZER
        with self.assertRaises(AssertionError):
            Parameterizer(self.topology, self.mol_specs, self.box, self.force_fields,
                          parameterizer_type=self.parameterizer_type,
                          parameterizer_assignment=ParameterizerAssignment.EXPLICIT,
                          customize_force_field=self.customize_force_field,
                          custom_file_paths=self.custom_file_paths)

    def test_assign_parameterizer(self):
        parameterizer = Parameterizer(self.topology, self.mol_specs, self.box, self.force_fields,
                                      parameterizer_type=self.parameterizer_type,
                                      parameterizer_assignment=self.parameterizer_assignment,
                                      customize_force_field=self.customize_force_field,
                                      custom_file_paths=self.custom_file_paths)

        parameterizer.assign_parameterizer(self.force_fields, self.parameterizer_type,
                                            ParameterizerAssignment.INFERRED)
        self.assertEqual(parameterizer.parameterizer_type, ParameterizerType.INTERCHANGE_PARAMETERIZER)

        parameterizer.assign_parameterizer(self.force_fields, self.parameterizer_type,
                                            ParameterizerAssignment.EXPLICIT)
        self.assertEqual(parameterizer.parameterizer_type, ParameterizerType.OPENMM_PARAMETERIZER)

        parameterizer.assign_parameterizer(self.force_fields, self.parameterizer_type,
                                            ParameterizerAssignment.DEFAULT)
        self.assertEqual(parameterizer.parameterizer_type, ParameterizerType.DEFAULT_PARAMETERIZER)

    def test_assign_topology(self):
        parameterizer = Parameterizer(self.topology, self.mol_specs, self.box, self.force_fields,
                                      parameterizer_type=self.parameterizer_type,
                                      parameterizer_assignment=self.parameterizer_assignment,
                                      customize_force_field=self.customize_force_field,
                                      custom_file_paths=self.custom_file_paths)

        self.assertEqual(parameterizer.topology, self.topology)

        new_topology = tk.Topology.from_molecules([tk.Molecule.from_smiles("CCCCC")])
        parameterizer.assign_topology(new_topology)
        self.assertEqual(parameterizer.topology, new_topology)



if __name__ == "__main__":
    unittest.main()
