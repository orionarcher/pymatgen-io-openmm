# type: ignore

import numpy as np
import parmed
import pytest
import openff.toolkit.topology
import pymatgen
import openmm
from openff.toolkit.typing.engines import smirnoff

# openmm
from openmm.unit import elementary_charge
from openmm.openmm import NonbondedForce

from pymatgen.io.openmm.utils import (
    smile_to_molecule,
    smile_to_parmed_structure,
    n_mols_from_mass_ratio,
    n_mols_from_volume_ratio,
    n_solute_from_molarity,
    calculate_molarity,
    order_molecule_canonically,
    add_mol_charges_to_forcefield,
    assign_charges_to_mols,
    parameterize_system,
    get_openmm_topology,
)

from pymatgen.io.openmm.sets import OpenMMSet
from pymatgen.io.openmm.generators import OpenMMSolutionGen

from pymatgen.io.openmm.tests.datafiles import (
    PF6_xyz,
    CCO_xyz,
    CCO_charges,
    FEC_s_xyz,
    FEC_charges,
    Li_charges,
    PF6_charges,
)


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


def test_n_mols_from_mass_ratio():
    n_mols = n_mols_from_mass_ratio(600, ["O", "CCO"], [10, 1])
    np.testing.assert_allclose(n_mols, [577, 23])


def test_n_mols_from_volume_ratio():
    n_mols = n_mols_from_volume_ratio(600, ["O", "CCO"], [10, 1], [1, 0.79])
    np.testing.assert_allclose(n_mols, [582, 18])


def test_n_solute_from_molarity():
    nm3_to_L = 1e-24
    n_solute = n_solute_from_molarity(1, 2**3 * nm3_to_L)
    np.testing.assert_allclose(n_solute, 5)
    n_solute = n_solute_from_molarity(1, 4**3 * nm3_to_L)
    np.testing.assert_allclose(n_solute, 39)


def test_calculate_molarity():
    nm3_to_L = 1e-24
    np.testing.assert_almost_equal(
        calculate_molarity(4**3 * nm3_to_L, 39), 1, decimal=1
    )


@pytest.mark.parametrize(
    "xyz_path, smile, atomic_numbers",
    [
        (CCO_xyz, "CCO", (6, 6, 8, 1, 1, 1, 1, 1, 1)),
        (PF6_xyz, "F[P-](F)(F)(F)(F)F", (9, 15, 9, 9, 9, 9, 9)),
    ],
)
def test_order_molecule_canonically(xyz_path, smile, atomic_numbers):
    mol = pymatgen.core.Molecule.from_file(xyz_path)
    ordered_mol = order_molecule_canonically(smile, mol)
    np.testing.assert_almost_equal(ordered_mol.atomic_numbers, atomic_numbers)


def test_get_openmm_topology():
    topology = get_openmm_topology({"O": 200, "CCO": 20})
    assert isinstance(topology, openmm.app.Topology)
    assert topology.getNumAtoms() == 780
    assert topology.getNumResidues() == 220
    assert topology.getNumBonds() == 560
    ethanol_smile = "CCO"
    fec_smile = "O=C1OC[C@H](F)O1"
    topology = get_openmm_topology({ethanol_smile: 50, fec_smile: 50})
    assert topology.getNumAtoms() == 950


def test_assign_charges_to_mols():
    # set up partial charges
    ethanol_mol = pymatgen.core.Molecule.from_file(CCO_xyz)
    fec_mol = pymatgen.core.Molecule.from_file(FEC_s_xyz)
    ethanol_charges = np.load(CCO_charges)
    fec_charges = np.load(FEC_charges)
    partial_charges = [(ethanol_mol, ethanol_charges), (fec_mol, fec_charges)]
    # set up force field
    ethanol_smile = "CCO"
    fec_smile = "O=C1OC[C@H](F)O1"
    li_smile = "[Li+]"
    openff_forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
    charged_mols = assign_charges_to_mols(
        [ethanol_smile, fec_smile, li_smile],
        "am1bcc",
        {},
        partial_charges,
    )
    openff_forcefield_scaled = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
    charged_mols_scaled = assign_charges_to_mols(
        [ethanol_smile, fec_smile, li_smile],
        "am1bcc",
        {ethanol_smile: 0.9, fec_smile: 0.9, li_smile: 0.9},
        partial_charges,
    )
    # construct a System to make testing easier
    topology = get_openmm_topology({ethanol_smile: 50, fec_smile: 50, li_smile: 1})
    openff_topology = openff.toolkit.topology.Topology.from_openmm(
        topology, charged_mols
    )
    openff_topology_scaled = openff.toolkit.topology.Topology.from_openmm(
        topology, charged_mols_scaled
    )
    system = openff_forcefield.create_openmm_system(
        openff_topology,
        charge_from_molecules=charged_mols,
    )
    system_scaled = openff_forcefield_scaled.create_openmm_system(
        openff_topology_scaled,
        charge_from_molecules=charged_mols_scaled,
        allow_nonintegral_charges=True,
    )
    # ensure that all forces are from our assigned force field
    # this does not ensure correct ordering, as we already test that with
    # other methods
    fec_charges_reordered = fec_charges[[0, 1, 2, 3, 4, 6, 7, 8, 9, 5]]
    full_partial_array = np.append(
        np.tile(ethanol_charges, 50), np.tile(fec_charges_reordered, 50)
    )
    full_partial_array = np.append(full_partial_array, np.array([1]))
    for force in system.getForces():
        if type(force) == NonbondedForce:
            charge_array = np.zeros(force.getNumParticles())
            for i in range(len(charge_array)):
                charge_array[i] = force.getParticleParameters(i)[0]._value
    np.testing.assert_allclose(full_partial_array, charge_array, atol=0.0001)
    for force in system_scaled.getForces():
        if type(force) == NonbondedForce:
            charge_array = np.zeros(force.getNumParticles())
            for i in range(len(charge_array)):
                charge_array[i] = force.getParticleParameters(i)[0]._value
    np.testing.assert_allclose(full_partial_array * 0.9, charge_array, atol=0.0001)


def test_parameterize_system():
    # TODO: add test here to see if I am adding charges?
    topology = get_openmm_topology({"O": 200, "CCO": 20})
    smile_strings = ["O", "CCO"]
    box = [0, 0, 0, 19.59, 19.59, 19.59]
    force_field = "Sage"
    partial_charge_method = "am1bcc"
    system = parameterize_system(
        topology,
        smile_strings,
        box,
        force_field=force_field,
        partial_charge_method=partial_charge_method,
        partial_charge_scaling={},
        partial_charges=[],
    )
    assert system.getNumParticles() == 780
    assert system.usesPeriodicBoundaryConditions()


@pytest.mark.parametrize(
    "charges_path, smile, atom_values",
    [
        (Li_charges, "[Li+]", [0]),
        (CCO_charges, "CCO", [0, 1, 2, 3, 4, 5, 6, 7, 8]),
        (FEC_charges, "O=C1OC[C@@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 8, 9, 5]),
        (FEC_charges, "O=C1OC[C@H](F)O1", [0, 1, 2, 3, 4, 6, 7, 8, 9, 5]),
        (PF6_charges, "F[P-](F)(F)(F)(F)F", [1, 0, 2, 3, 4, 5, 6]),
    ],
)
def test_add_mol_charges_to_forcefield(charges_path, smile, atom_values):
    charges = np.load(charges_path)
    openff_mol = openff.toolkit.topology.Molecule.from_smiles(smile)
    atom_map = {i: j for i, j in enumerate(atom_values)}  # this saves some space
    mapped_charges = np.array([charges[atom_map[i]] for i in range(len(charges))])
    openff_mol.partial_charges = mapped_charges * elementary_charge
    forcefield = smirnoff.ForceField("openff_unconstrained-2.0.0.offxml")
    add_mol_charges_to_forcefield(forcefield, openff_mol)
    topology = openff_mol.to_topology()
    system = forcefield.create_openmm_system(topology)
    for force in system.getForces():
        if type(force) == NonbondedForce:
            expected = np.array(
                [
                    force.getParticleParameters(i)[0]._value
                    for i in range(force.getNumParticles())
                ]
            )
            np.testing.assert_allclose(expected, mapped_charges, atol=0.01)


# TODO: refactor to test new parameterize system
@pytest.mark.parametrize(
    "w_ff, sm_ff", [("spce", "gaff"), ("spce", "sage"), ("tip3p", "gaff")]
)
def test_parameterize_mixed_forcefield_system(w_ff, sm_ff):
    # TODO: test with charges
    # TODO: periodic boundaries assertion
    # TODO: assert forcefields assigned correctly
    topology = get_openmm_topology({"O": 200, "CCO": 20})
    smile_strings = ["O", "CCO"]
    box = [0, 0, 0, 19.59, 19.59, 19.59]
    force_field = {"O": w_ff, "CCO": sm_ff}
    partial_charge_method = "am1bcc"
    system = parameterize_system(
        topology,
        smile_strings,
        box,
        force_field=force_field,
        partial_charge_method=partial_charge_method,
        partial_charge_scaling={},
        partial_charges=[],
    )
    assert len(system.getForces()) > 0
    wforce = system.getForces()[0].getBondParameters(0)
    notwforce = system.getForces()[0].getBondParameters(401)
    assert wforce != notwforce
    assert system.getNumParticles() == 780
    assert system.usesPeriodicBoundaryConditions()


# TODO refactor, perhaps disable for now
@pytest.mark.parametrize(
    "modela, modelb", [("spce", "tip3p"), ("amber14/tip3p.xml", "amber14/tip3pfb.xml")]
)
def test_water_models(modela, modelb):
    topology = get_openmm_topology({"O": 200})
    smile_strings = ["O"]
    box = [0, 0, 0, 19.59, 19.59, 19.59]
    force_field_a = {"O": modela}
    partial_charge_method = "am1bcc"
    system_a = parameterize_system(
        topology,
        smile_strings,
        box,
        force_field=force_field_a,
        partial_charge_method=partial_charge_method,
        partial_charge_scaling={},
        partial_charges=[],
    )
    force_field_b = {"O": modelb}
    partial_charge_method = "am1bcc"
    system_b = parameterize_system(
        topology,
        smile_strings,
        box,
        force_field=force_field_b,
        partial_charge_method=partial_charge_method,
        partial_charge_scaling={},
        partial_charges=[],
    )
    force_a = system_a.getForces()[0].getBondParameters(0)
    force_b = system_b.getForces()[0].getBondParameters(0)
    # assert rOH is different for two different water models
    assert force_a[2] != force_b[2]

    def test_get_input_set_w_charges_and_forcefields(self):
        np.load(PF6_charges)
        np.load(Li_charges)
        generator = OpenMMSolutionGen(
            packmol_random_seed=1,
        )
        input_set = generator.get_input_set(
            {"O": 200, "CCO": 20, "F[P-](F)(F)(F)(F)F": 10, "[Li+]": 10}, density=1
        )
        assert isinstance(input_set, OpenMMSet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
        }
        assert input_set.validate()

    def test_formal_charge(self):
        trimer_smile = (
            "O=C1[C@H]([C@H](OC(O)=C1/C(CCCCCCCC/C(O[H])=C2C(["
            "C@H]([C@H](OC/2=O)C(C)C)C)=O)=[NH+]/CCN(CC/[NH+]=C("
            r"C3=C(O[C@@H]([C@@H](C3=O)C)C(C)C)O)\CCCCCCCC/C(O["
            "H])=C4C([C@H]([C@H](OC/4=O)C(C)C)C)=O)CC/[NH+]=C("
            r"C5=C(O[C@@H]([C@@H](C5=O)C)C(C)C)O)\CCCCCCCC/C(O["
            "H])=C6C([C@H]([C@H](OC/6=O)C(C)C)C)=O)C(C)C)C "
        )
        openmm_generator = OpenMMSolutionGen(
            temperature=298,
            step_size=0.001,
            partial_charge_method="mmff94",
            force_field={"O": "spce", trimer_smile: "sage"},
        )

        molecules = {"O": 200, trimer_smile: 1}

        input_set = openmm_generator.get_input_set(molecules, density=0.5)

        assert isinstance(input_set, OpenMMSet)
        assert set(input_set.inputs.keys()) == {
            "topology.pdb",
            "system.xml",
            "integrator.xml",
            "state.xml",
        }
        assert input_set.validate()
