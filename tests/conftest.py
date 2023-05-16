import math
import itertools
import numpy as np
import sparse
import pytest
import scipy
import scipy.stats
from scipy.signal import find_peaks
from ase import Atoms
from ase.build import molecule, bulk
from dscribe.descriptors.descriptorlocal import DescriptorLocal

"""
Contains a set of shared test functions.
"""
setup_k1 = {
    "geometry": {"function": "atomic_number"},
    "grid": {"min": 1, "max": 8, "sigma": 0.1, "n": 50},
    "weighting": {"function": "unity"},
}

setup_k2 = {
    "geometry": {"function": "inverse_distance"},
    "grid": {"min": 0, "max": 1 / 0.7, "sigma": 0.1, "n": 50},
    "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-2},
}

setup_k3 = {
    "geometry": {"function": "angle"},
    "grid": {"min": 0, "max": 180, "sigma": 2, "n": 50},
    "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-2},
}


def get_complex_periodic():
    """Elaborate test system with multiple species, non-cubic cell, and close-by
    atoms.
    """
    a = 1
    return Atoms(
        symbols=["C", "C", "C"],
        cell=[[0, a, a], [a, 0, a], [a, a, 0]],
        scaled_positions=[
            [0, 0, 0],
            [1 / 3, 1 / 3, 1 / 3],
            [2 / 3, 2 / 3, 2 / 3],
        ],
        pbc=[True, True, True],
    ) * (3, 3, 3)


def get_simple_periodic():
    """Simple bulk system."""
    return Atoms(
        cell=[5, 5, 5],
        scaled_positions=[
            [0.1, 0, 0],
            [0.9, 0, 0],
        ],
        symbols=["H", "H"],
        pbc=True,
    )


def get_simple_finite():
    """The H2O molecule in a cell."""
    return Atoms(
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        positions=[
            [0, 0, 0],
            [0.95, 0, 0],
            [
                0.95 * (1 + math.cos(76 / 180 * math.pi)),
                0.95 * math.sin(76 / 180 * math.pi),
                0.0,
            ],
        ],
        symbols=["H", "O", "H"],
    )


def get_complex_finite():
    """Acetyl fluoride molecule in a cell with no periodicity."""
    mol = molecule("CH3COF")
    mol.set_cell([5, 5, 5])
    mol.center()
    return mol


def assert_symmetries(
    descriptor_func, translation=True, rotation=True, permutation=True
):
    if translation:
        assert_symmetry_translation(descriptor_func)
    if rotation:
        assert_symmetry_rotation(descriptor_func)
    if permutation:
        assert_symmetry_permutation(descriptor_func)


def assert_symmetry_rotation(descriptor_func):
    """Tests whether the descriptor output is invariant to rotations of the
    original system.
    """
    system = get_simple_finite()
    descriptor = descriptor_func([system])
    features = descriptor.create(system)
    is_rot_sym = True

    # Rotational Check
    for rotation in ["x", "y", "z"]:
        i_system = system.copy()
        i_system.rotate(45, rotation, rotate_cell=True)
        i_features = descriptor.create(i_system)
        deviation = np.max(np.abs(features - i_features))
        if deviation > 1e-4:
            is_rot_sym = False
    assert is_rot_sym


def assert_symmetry_translation(descriptor_func):
    """Tests whether the descriptor output is invariant to translations of
    the original system.

    Args:
        create(function): A function that when given an Atoms object
        returns a final descriptor vector for it.
    """
    system = get_simple_finite()
    descriptor = descriptor_func([system])
    features = descriptor.create(system)
    is_trans_sym = True

    # Rotational Check
    for translation in [[1.0, 1.0, 1.0], [-5.0, 5.0, -5.0], [1.0, 1.0, -10.0]]:
        i_system = system.copy()
        i_system.translate(translation)
        i_features = descriptor.create(i_system)
        deviation = np.max(np.abs(features - i_features))
        if deviation > 1e-4:
            is_trans_sym = False

    assert is_trans_sym


def assert_symmetry_permutation(descriptor_func):
    """Tests whether the descriptor output is invariant to permutation of
    atom indexing.
    """
    system = get_simple_finite()
    descriptor = descriptor_func([system])
    features = descriptor.create(system)
    is_perm_sym = True

    for permutation in ([0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]):
        i_system = system[permutation]
        i_features = descriptor.create(i_system)
        deviation = np.max(np.abs(features - i_features))
        if deviation > 1e-7:
            is_perm_sym = False

    assert is_perm_sym


def assert_periodic(descriptor_func):
    """Tests the periodic images are correctly seen by the descriptor."""
    a = 10
    reference = Atoms(
        symbols=["C", "C"],
        positions=[[0, 0, 0], [-1, -1, -1]],
        cell=[[0.0, a, a], [a, 0.0, a], [a, a, 0.0]],
        pbc=[False],
    )

    periodic = reference.copy()
    periodic.set_pbc(True)
    periodic.wrap()

    desc = descriptor_func(periodic=True)([reference])
    feat_ref = desc.create(reference)
    feat_periodic = desc.create(periodic)
    assert feat_ref.max() > 0.5
    assert np.allclose(feat_ref, feat_periodic)


def assert_derivatives(
    descriptor_func,
    method,
    pbc,
    system=get_complex_periodic(),
    attach=None,
    create_args=None,
):
    # Test values against a naive python implementation.
    system.set_pbc(pbc)
    h = 0.0001
    n_atoms = len(system)
    n_comp = 3
    descriptor = descriptor_func([system])
    n_features = descriptor.get_number_of_features()
    coeffs = [-1.0 / 2.0, 1.0 / 2.0]
    deltas = [-1.0, 1.0]
    if create_args is None:
        create_args = {}

    if isinstance(descriptor, DescriptorLocal):
        if attach:
            centers = [38, 0]
        else:
            centers = [np.sum(system.get_cell(), axis=0) / 2, system.get_positions()[0]]
        average = descriptor.average
        if average != "off":
            n_centers = 1
            derivatives_python = np.zeros((n_atoms, n_comp, n_features))
        else:
            n_centers = len(centers)
            derivatives_python = np.zeros((n_centers, n_atoms, n_comp, n_features))
        d0 = descriptor.create(system, centers, **create_args)
        for i_atom in range(len(system)):
            for i_center in range(n_centers):
                for i_comp in range(3):
                    for i_stencil in range(2):
                        if average == "off":
                            i_cent = [centers[i_center]]
                        else:
                            i_cent = centers
                        system_disturbed = system.copy()
                        i_pos = system_disturbed.get_positions()
                        i_pos[i_atom, i_comp] += h * deltas[i_stencil]
                        system_disturbed.set_positions(i_pos)
                        d1 = descriptor.create(system_disturbed, i_cent, **create_args)
                        if average != "off":
                            derivatives_python[i_atom, i_comp, :] += (
                                coeffs[i_stencil] * d1 / h
                            )
                        else:
                            derivatives_python[i_center, i_atom, i_comp, :] += (
                                coeffs[i_stencil] * d1[0, :] / h
                            )
        derivatives_cpp, d_cpp = descriptor.derivatives(
            system, centers=centers, attach=attach, method=method
        )
    else:
        derivatives_python = np.zeros((n_atoms, n_comp, n_features))
        d0 = descriptor.create(system, **create_args)
        for i_atom in range(len(system)):
            for i_comp in range(3):
                for i_stencil in range(2):
                    system_disturbed = system.copy()
                    i_pos = system_disturbed.get_positions()
                    i_pos[i_atom, i_comp] += h * deltas[i_stencil]
                    system_disturbed.set_positions(i_pos)
                    d1 = descriptor.create(system_disturbed, **create_args)
                    derivatives_python[i_atom, i_comp, :] += coeffs[i_stencil] * d1 / h
        derivatives_cpp, d_cpp = descriptor.derivatives(system, method=method)

    # The maximum error depends on how big the system is. With a small
    # system the error is smaller for non-periodic systems than the
    # corresponding error when periodicity is turned on. The errors become
    # equal when the size of the system is increased.

    # Compare descriptor values
    assert np.allclose(d0, d_cpp, atol=1e-6)

    # Check that derivatives are not super small: this typically indicates other
    # problems.
    assert np.max(np.abs(derivatives_cpp)) > 1e-8

    # Compare derivative values
    assert np.allclose(derivatives_python, derivatives_cpp, rtol=0.5e-3, atol=1e-4)


def assert_derivatives_include(descriptor_func, method, attach=None):
    H2O = molecule("H2O")
    CO2 = molecule("CO2")
    H2O.set_cell([5, 5, 5])
    CO2.set_cell([5, 5, 5])
    descriptor = descriptor_func([H2O, CO2])
    kwargs = {"method": method}

    if isinstance(descriptor, DescriptorLocal):
        kwargs["attach"] = attach
        slice_d1_a = np.index_exp[:, 0, :]
        slice_d2_a = np.index_exp[:, 2, :]
        slice_d1_b = np.index_exp[:, 1, :]
        slice_d2_b = np.index_exp[:, 0, :]
        slice_d1_c = np.index_exp[:, :, 0, :]
        slice_d2_c = np.index_exp[:, :, 1, :]
        slice_d1_d = np.index_exp[:, :, 1, :]
        slice_d2_d = np.index_exp[:, :, 0, :]
        slice_d1_e = np.index_exp[0, :, 0, :]
        slice_d2_e = np.index_exp[0, :, 0, :]
        slice_d1_f = np.index_exp[1, :, 0, :]
        slice_d2_f = np.index_exp[1, :, 1, :]
    else:
        slice_d1_a = np.index_exp[0, :]
        slice_d2_a = np.index_exp[2, :]
        slice_d1_b = np.index_exp[1, :]
        slice_d2_b = np.index_exp[0, :]
        slice_d1_c = np.index_exp[:, 0, :]
        slice_d2_c = np.index_exp[:, 1, :]
        slice_d1_d = np.index_exp[:, 1, :]
        slice_d2_d = np.index_exp[:, 0, :]
        slice_d1_e = np.index_exp[0, 0, :]
        slice_d2_e = np.index_exp[0, 0, :]
        slice_d1_f = np.index_exp[1, 0, :]
        slice_d2_f = np.index_exp[1, 1, :]

    # Invalid include options
    with pytest.raises(ValueError):
        descriptor.derivatives(H2O, include=[], **kwargs)
    with pytest.raises(ValueError):
        descriptor.derivatives(H2O, include=[3], **kwargs)
    with pytest.raises(ValueError):
        descriptor.derivatives(H2O, include=[-1], **kwargs)
    with pytest.raises(ValueError):
        descriptor.derivatives(H2O, include=[0], exclude=[0], **kwargs)

    # Test that correct atoms are included and in the correct order
    D1, d1 = descriptor.derivatives(H2O, include=[2, 0], **kwargs)
    D2, d2 = descriptor.derivatives(H2O, **kwargs)
    assert np.array_equal(D1[slice_d1_a], D2[slice_d2_a])
    assert np.array_equal(D1[slice_d1_b], D2[slice_d2_b])

    # Test that using multiple samples and single include works
    D1, d1 = descriptor.derivatives([H2O, CO2], include=[1, 0], **kwargs)
    D2, d2 = descriptor.derivatives([H2O, CO2], **kwargs)
    assert np.array_equal(D1[slice_d1_c], D2[slice_d2_c])
    assert np.array_equal(D1[slice_d1_d], D2[slice_d2_d])

    # Test that using multiple samples and multiple includes
    D1, d1 = descriptor.derivatives([H2O, CO2], include=[[0], [1]], **kwargs)
    D2, d2 = descriptor.derivatives([H2O, CO2], **kwargs)
    assert np.array_equal(D1[slice_d1_e], D2[slice_d2_e])
    assert np.array_equal(D1[slice_d1_f], D2[slice_d2_f])


def assert_derivatives_exclude(descriptor_func, method, attach=None):
    H2O = molecule("H2O")
    CO2 = molecule("CO2")
    H2O.set_cell([5, 5, 5])
    CO2.set_cell([5, 5, 5])
    descriptor = descriptor_func([H2O, CO2])
    kwargs = {"method": method}

    if isinstance(descriptor, DescriptorLocal):
        kwargs["attach"] = attach
        slice_d1_a = np.index_exp[:, 0, :]
        slice_d2_a = np.index_exp[:, 0, :]
        slice_d1_b = np.index_exp[:, 1, :]
        slice_d2_b = np.index_exp[:, 2, :]
        slice_d1_c = np.index_exp[:, :, 0, :]
        slice_d2_c = np.index_exp[:, :, 0, :]
        slice_d1_d = np.index_exp[:, :, 1, :]
        slice_d2_d = np.index_exp[:, :, 2, :]
    else:
        slice_d1_a = np.index_exp[0, :]
        slice_d2_a = np.index_exp[0, :]
        slice_d1_b = np.index_exp[1, :]
        slice_d2_b = np.index_exp[2, :]
        slice_d1_c = np.index_exp[:, 0, :]
        slice_d2_c = np.index_exp[:, 0, :]
        slice_d1_d = np.index_exp[:, 1, :]
        slice_d2_d = np.index_exp[:, 2, :]

    # Invalid exclude options
    with pytest.raises(ValueError):
        descriptor.derivatives(H2O, exclude=[3], **kwargs)
    with pytest.raises(ValueError):
        descriptor.derivatives(H2O, exclude=[-1], **kwargs)

    # Test that correct atoms are excluded and in the correct order
    D1, d1 = descriptor.derivatives(H2O, exclude=[1], **kwargs)
    D2, d2 = descriptor.derivatives(H2O, **kwargs)
    assert np.array_equal(D1[slice_d1_a], D2[slice_d2_a])
    assert np.array_equal(D1[slice_d1_b], D2[slice_d2_b])

    # Test that using single list and multiple samples works
    D1, d1 = descriptor.derivatives([H2O, CO2], exclude=[1], **kwargs)
    D2, d2 = descriptor.derivatives([H2O, CO2], **kwargs)
    assert np.array_equal(D1[slice_d1_c], D2[slice_d2_c])
    assert np.array_equal(D1[slice_d1_d], D2[slice_d2_d])


def assert_cell(descriptor_func, cell):
    """Test that different types of cells are handled correctly."""
    system = get_simple_finite()
    # Error is raised if there is no cell when periodicity=True
    if cell == "collapsed_periodic":
        system.set_pbc(True)
        system.set_cell([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        desc = descriptor_func(periodic=True)([system])
        with pytest.raises(ValueError):
            desc.create(system)
    # No error is raised if there is no cell when periodicity=False
    elif cell == "collapsed_finite":
        system.set_cell([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        desc = descriptor_func(periodic=False)([system])
        desc.create(system)
    else:
        raise ValueError("Unknown cell option")


def assert_systems(descriptor_func, pbc, cell):
    """Tests that the descriptor can correctly handle differently described
    systems
    """
    system = get_simple_finite()
    system.set_pbc(pbc)
    system.set_cell(cell)
    descriptor = descriptor_func([system])
    descriptor.create(system)


def assert_dtype(descriptor_func, dtype, sparse):
    """Test that the requested data type is respected"""
    system = get_simple_finite()
    desc = descriptor_func(dtype=dtype, sparse=sparse)([system])
    features = desc.create(get_simple_finite())
    assert features.dtype == dtype


def assert_sparse(descriptor_func):
    """Test that sparse output is created upon request in the correct format."""
    system = get_simple_finite()

    # Dense
    desc = descriptor_func(sparse=False)([system])
    features = desc.create(system)
    assert type(features) == np.ndarray

    # Sparse
    desc = descriptor_func(sparse=True)([system])
    features = desc.create(system)
    assert type(features) == sparse.COO


def assert_n_features(descriptor_func, n_features):
    """Test that the reported number of features matches the actual number of
    features and the expected value.
    """
    system = get_simple_finite()
    desc = descriptor_func([system])
    n_features_reported = desc.get_number_of_features()
    n_features_actual = desc.create(system).shape[-1]
    assert n_features_reported == n_features_actual == n_features


def assert_parallellization(descriptor_func, n_jobs, sparse, centers=None, **kwargs):
    """Tests creating output and derivatives parallelly."""
    # Periodic systems are used since all descriptors support them.
    CO = molecule("CO")
    CO.set_cell([5, 5, 5])
    CO.set_pbc(True)
    NO = molecule("NO")
    NO.set_cell([5, 5, 5])
    NO.set_pbc(True)
    samples = [CO, NO]
    kwargs["sparse"] = sparse
    desc = descriptor_func(**kwargs)(samples)

    for func in ["create", "derivatives"]:
        all_kwargs = {}
        a_kwargs = {}
        b_kwargs = {}
        if centers == "all":
            all_kwargs["centers"] = None
            a_kwargs["centers"] = None
            b_kwargs["centers"] = None
        elif centers == "indices_fixed":
            all_kwargs["centers"] = [[0, 1], [0, 1]]
            a_kwargs["centers"] = [0, 1]
            b_kwargs["centers"] = [0, 1]
        elif centers == "indices_variable":
            all_kwargs["centers"] = [[0], [0, 1]]
            a_kwargs["centers"] = [0]
            b_kwargs["centers"] = [0, 1]
        elif centers == "cartesian_fixed":
            all_kwargs["centers"] = [[[0, 0, 0], [1, 2, 0]], [[0, 0, 0], [1, 2, 0]]]
            a_kwargs["centers"] = [[0, 0, 0], [1, 2, 0]]
            b_kwargs["centers"] = [[0, 0, 0], [1, 2, 0]]
        elif centers == "cartesian_variable":
            all_kwargs["centers"] = [[[1, 2, 0]], [[0, 0, 0], [1, 2, 0]]]
            a_kwargs["centers"] = [[1, 2, 0]]
            b_kwargs["centers"] = [[0, 0, 0], [1, 2, 0]]
        elif centers is not None:
            raise ValueError("Unknown centers option")

        if func == "create":
            output = desc.create(samples, n_jobs=n_jobs, **all_kwargs)
            a = desc.create(samples[0], **a_kwargs)
            b = desc.create(samples[1], **b_kwargs)
        elif func == "derivatives":
            if isinstance(desc, DescriptorLocal):
                all_kwargs["attach"] = True
                a_kwargs["attach"] = True
                b_kwargs["attach"] = True
            output, _ = desc.derivatives(samples, n_jobs=n_jobs, **all_kwargs)
            a, _ = desc.derivatives(samples[0], **a_kwargs)
            b, _ = desc.derivatives(samples[1], **b_kwargs)

        # The output may be a list or an array.
        if isinstance(output, list):
            for i, j in zip(output, [a, b]):
                if sparse:
                    i = i.todense()
                    j = j.todense()
                assert np.allclose(i, j)
        else:
            if sparse:
                a = a.todense()
                b = b.todense()
                output = output.todense()
            assumed = np.array([a, b])
            assert np.allclose(output, assumed)


def assert_no_system_modification(descriptor_func):
    """Tests that the descriptor does not modify the system that is given as
    input.
    """

    def check_modifications(system):
        cell = np.array(system.get_cell())
        pos = np.array(system.get_positions())
        pbc = np.array(system.get_pbc())
        atomic_numbers = np.array(system.get_atomic_numbers())
        symbols = np.array(system.get_chemical_symbols())
        descriptor_func()([system])
        assert np.array_equal(cell, system.get_cell())
        assert np.array_equal(pos, system.get_positions())
        assert np.array_equal(pbc, system.get_pbc())
        assert np.array_equal(atomic_numbers, system.get_atomic_numbers())
        assert np.array_equal(symbols, system.get_chemical_symbols())

    # Try separately for periodic and non-periodic systems
    check_modifications(bulk("Cu", "fcc", a=3.6))
    check_modifications(get_simple_finite())


def assert_basis(descriptor_func):
    """Tests that the output vectors behave correctly as a basis."""
    sys1 = Atoms(
        symbols=["H", "O"], positions=[[0, 0, 0], [1, 1, 1]], cell=[2, 2, 2], pbc=True
    )
    sys2 = Atoms(
        symbols=["O", "C"], positions=[[0, 0, 0], [1, 1, 1]], cell=[2, 2, 2], pbc=True
    )
    sys3 = Atoms(
        symbols=["C", "N"], positions=[[0, 0, 0], [1, 1, 1]], cell=[2, 2, 2], pbc=True
    )
    sys4 = sys3 * [2, 2, 2]

    desc = descriptor_func([sys1, sys2, sys3, sys4])

    # Create vectors for each system
    vec1 = desc.create(sys1)
    vec2 = desc.create(sys2)
    vec3 = desc.create(sys3)
    vec4 = desc.create(sys4)

    # Average the result for local descriptors
    if isinstance(desc, DescriptorLocal):
        vec1 = np.mean(vec1, 0)
        vec2 = np.mean(vec2, 0)
        vec3 = np.mean(vec3, 0)
        vec4 = np.mean(vec4, 0)

    # Create normalized vectors
    vec1 /= np.linalg.norm(vec1)
    vec2 /= np.linalg.norm(vec2)
    vec3 /= np.linalg.norm(vec3)
    vec4 /= np.linalg.norm(vec4)

    # import matplotlib.pyplot as mpl
    # mpl.plot(vec2)
    # mpl.plot(vec3)
    # mpl.show()

    # The dot-product should be zero when there are no overlapping elements
    dot = np.dot(vec1, vec3)
    assert dot == 0

    # The dot-product should be non-zero when there are overlapping elements
    dot = np.dot(vec1, vec2)
    assert dot > 0 and dot < 1

    # After normalization, the dot-product should be roughly one for a primitive
    # cell and a supercell
    dot = np.dot(vec3, vec4)
    assert abs(dot - 1) < 1e-3


def assert_normalization(
    descriptor_func, system, normalization, norm_rel=None, norm_abs=None
):
    """Tests that the normalization of the output is done correctly."""
    desc_raw = descriptor_func(normalization="none", periodic=True)([system])
    features_raw = desc_raw.create(system)
    desc_normalized = descriptor_func(normalization=normalization, periodic=True)(
        [system]
    )
    features_normalized = desc_normalized.create(system)
    is_local = isinstance(desc_normalized, DescriptorLocal)
    if is_local:
        features_raw = features_raw[0]
        features_normalized = features_normalized[0]

    norm = np.linalg.norm(features_normalized)
    if norm_rel is not None and norm_abs is not None:
        raise ValueError("Provide only relative or absolute norm")
    if norm_rel is not None:
        norm_raw = np.linalg.norm(features_raw)
        assert norm_raw != 0
        assert norm / norm_raw == pytest.approx(norm_rel, 0, 1e-8)
    if norm_abs is not None:
        assert norm == pytest.approx(norm_abs, 0, 1e-8)


def assert_centers(descriptor_func):
    """Tests that local descriptors handle the centers argument correctly."""
    system = get_simple_finite()
    desc = descriptor_func()([system])
    centers = [
        ([0, 1, 2], 3),
        ([[0, 1, 2], [0, 1, 1]], 2),
        (np.array([[0, 1, 2], [0, 1, 1]]), 2),
        (("a"), 0),
        ((3), 0),
        ((-1), 0),
    ]
    for pos, n_pos in centers:
        if not n_pos:
            with pytest.raises(Exception):
                desc.create(system, centers=pos)
        else:
            feat = desc.create(system, centers=pos)
            assert feat.shape[0] == n_pos


def assert_matrix_descriptor_exceptions(descriptor_func):
    system = get_simple_finite()

    # Unknown permutation option
    with pytest.raises(ValueError):
        descriptor_func(n_atoms_max=5, permutation="unknown")()
    # Negative n_atom_max
    with pytest.raises(ValueError):
        descriptor_func(n_atoms_max=-1)()
    # System has more atoms than the specified maximum
    with pytest.raises(ValueError):
        cm = descriptor_func(n_atoms_max=2)()
        cm.create([system])


def assert_matrix_descriptor_sorted(descriptor_func):
    """Tests that sorting using row norm works as expected"""
    system = get_complex_finite()
    desc = descriptor_func(permutation="sorted_l2")([system])
    features = desc.create(system)
    features = desc.unflatten(features)

    # Check that norms are ordered correctly
    lens = np.linalg.norm(features, axis=1)
    old_len = lens[0]
    for length in lens[1:]:
        assert length <= old_len
        old_len = length

    # Check that the matrix is symmetric
    assert np.allclose(features, features.T, rtol=0, atol=1e-13)


def assert_matrix_descriptor_eigenspectrum(descriptor_func):
    """Tests that the eigenvalues are sorted correctly and that the output is
    zero-padded.
    """
    system = get_simple_finite()
    desc = descriptor_func(n_atoms_max=5, permutation="eigenspectrum")([system])
    features = desc.create(system)

    assert features.shape == (5,)

    # Test that eigenvalues are in decreasing order when looking at absolute
    # value
    prev_eig = float("Inf")
    for eigenvalue in features[: len(system)]:
        assert abs(eigenvalue) <= abs(prev_eig)
        prev_eig = eigenvalue

    # Test that array is zero-padded
    assert np.array_equal(features[len(system) :], [0, 0])


def assert_matrix_descriptor_random(descriptor_func):
    """Tests if the random sorting obeys a gaussian distribution. Could
    possibly fail even though everything is OK.

    Measures how many times the two rows with biggest norm exchange place when
    random noise is added. This should correspond to the probability P(X > Y),
    where X = N(\\mu_1, \\sigma^2), Y = N(\\mu_2, \\sigma^2). This probability can
    be reduced to P(X > Y) = P(X-Y > 0) = P(N(\\mu_1 - \\mu_2, \\sigma^2 +
    \\sigma^2) > 0). See e.g.
    https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables
    """
    HHe = Atoms(
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        positions=[
            [0, 0, 0],
            [0.71, 0, 0],
        ],
        symbols=["H", "He"],
    )

    # Get the mean value to compare to
    sigma = 5
    desc = descriptor_func(permutation="sorted_l2")([HHe])
    features = desc.create(HHe)
    features = desc.unflatten(features)
    means = np.linalg.norm(features, axis=1)
    mu2 = means[0]
    mu1 = means[1]

    desc = descriptor_func(
        permutation="random",
        sigma=sigma,
        seed=42,
    )([HHe])
    count = 0
    rand_instances = 20000
    for i in range(0, rand_instances):
        features = desc.create(HHe)
        features = desc.unflatten(features)
        i_means = np.linalg.norm(features, axis=1)
        if i_means[0] < i_means[1]:
            count += 1

    # The expected probability is calculated from the cumulative
    # distribution function.
    expected = 1 - scipy.stats.norm.cdf(0, mu1 - mu2, np.sqrt(sigma**2 + sigma**2))
    observed = count / rand_instances

    assert abs(expected - observed) <= 1e-2


def assert_mbtr_location(descriptor_func, k):
    """Tests that the function used to query combination locations in the MBTR
    output works.
    """
    species = ["H", "O", "C"]
    systems = [molecule("CO2"), molecule("H2O")]

    setup = globals()[f"setup_k{k}"]
    desc = descriptor_func(**setup, periodic=False, species=species)([])
    is_local = isinstance(desc, DescriptorLocal)

    # The central atom species X is added for local MBTR flavors.
    if is_local:
        combinations = itertools.product(species + ["X"], repeat=k)
        combinations = list(filter(lambda x: "X" in x, combinations))
    else:
        combinations = list(itertools.product(species, repeat=k))

    for system in systems:
        feat = desc.create(system)
        symbols = system.get_chemical_symbols()
        if is_local:
            feat = feat.mean(axis=0)
            system_combinations = itertools.permutations(symbols + ["X"], k)
            system_combinations = list(filter(lambda x: "X" in x, system_combinations))
        else:
            system_combinations = list(itertools.permutations(symbols, k))

        for combination in combinations:
            loc = desc.get_location(combination)
            exists = combination in system_combinations
            combination_feats = feat[loc].sum()
            if exists:
                assert combination_feats != 0
            else:
                assert combination_feats == 0


def assert_mbtr_peak(
    descriptor_func, system, k, grid, geometry, weighting, periodic, peaks, prominence
):
    """Tests that the correct peaks are present in the descriptor output."""
    desc = descriptor_func(
        species=system.get_atomic_numbers(),
        grid=grid,
        geometry=geometry,
        weighting=weighting,
        periodic=periodic,
        normalize_gaussians=False,
        sparse=False,
    )([system])
    features = desc.create(system)
    if isinstance(desc, DescriptorLocal):
        features = features[0]

    start = grid["min"]
    stop = grid["max"]
    n = grid["n"]
    x = np.linspace(start, stop, n)

    # import matplotlib.pyplot as mpl
    # mpl.plot(features)
    # mpl.show()

    # Check that the correct peaks can be found
    for location, peak_x, peak_y in peaks:
        feat = features[desc.get_location(location)]
        peak_indices = find_peaks(feat, prominence=prominence)[0]
        assert len(peak_indices) > 0
        peak_locs = x[peak_indices]
        peak_ints = feat[peak_indices]
        assert np.allclose(peak_locs, peak_x, rtol=1e-3, atol=1e-3)
        assert np.allclose(peak_ints, peak_y, rtol=1e-3, atol=1e-3)

    # Check that everything else is zero
    for peak in peaks:
        features[desc.get_location(peak[0])] = 0
    assert features.sum() == 0


def assert_mbtr_location_exception(desc, location):
    """Test that an exception is raise when invalid location is requested."""
    with pytest.raises(ValueError):
        desc.get_location(location)
