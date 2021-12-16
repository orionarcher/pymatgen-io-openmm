import numpy as np
import parmed
import pymatgen

from pymatgen.io.openmm.utils import (
    get_box,
    smile_to_molecule,
    smile_to_parmed_structure,
    n_mols_from_mass_ratio,
    n_mols_from_volume_ratio,
    n_solute_from_molarity,
    calculate_molarity,
)


def test_get_box():
    box = get_box({"O": 200, "CCO": 20}, 1)
    assert isinstance(box, list)
    assert len(box) == 6
    np.testing.assert_allclose(box[0:3], 0, 2)
    np.testing.assert_allclose(box[3:6], 19.59, 2)


def test_smile_to_parmed_structure():
    struct1 = smile_to_parmed_structure("CCO")
    assert isinstance(struct1, parmed.Structure)
    assert len(struct1.atoms) == 9
    assert len(struct1.residues) == 1
    assert len(struct1.bonds) == 8
    struct2 = smile_to_parmed_structure("O")
    assert len(struct2.atoms) == 3
    assert len(struct2.residues) == 1
    assert len(struct2.bonds) == 2
    struct3 = smile_to_parmed_structure("O=C1OC[C@H](F)O1")
    assert len(struct3.atoms) == 10
    assert len(struct3.residues) == 1
    assert len(struct3.bonds) == 10


def test_smile_to_molecule():
    mol = smile_to_molecule("CCO")
    assert isinstance(mol, pymatgen.core.structure.Molecule)
    assert len(mol.sites) == 9


def test_counts_from_mass_ratio():
    n_mols = n_mols_from_mass_ratio(600, ["O", "CCO"], [10, 1])
    np.testing.assert_allclose(n_mols, [577, 23])


def test_n_mols_from_volume_ratio():
    n_mols = n_mols_from_volume_ratio(600, ["O", "CCO"], [10, 1], [1, 0.79])
    np.testing.assert_allclose(n_mols, [582, 18])


def test_n_solute_from_molarity():
    nm3_to_L = 1e-24
    n_solute = n_solute_from_molarity(1, 2 ** 3 * nm3_to_L)
    np.testing.assert_allclose(n_solute, 5)
    n_solute = n_solute_from_molarity(1, 4 ** 3 * nm3_to_L)
    np.testing.assert_allclose(n_solute, 39)


def test_calculate_molarity():
    nm3_to_L = 1e-24
    np.testing.assert_almost_equal(calculate_molarity(4 ** 3 * nm3_to_L, 39), 1, decimal=1)
