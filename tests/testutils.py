import numpy as np
import scipy
from pathlib import Path
from scipy.integrate import tplquad
from scipy.linalg import sqrtm
from ase import Atoms

# from pymatgen.analysis.ewald import EwaldSummation
# from pymatgen.core.structure import Structure
from dscribe.descriptors import SOAP
from joblib import Parallel, delayed
from conftest import get_simple_finite

folder = Path(__file__).parent


def get_soap_default_setup():
    """Returns an atomic system and SOAP parameters which are ideal for most
    tests.
    """
    # Calculate the numerical power spectrum
    system = Atoms(
        positions=[
            [0.0, 0.0, 0.0],
            [-0.3, 0.5, 0.4],
        ],
        symbols=["H", "C"],
    )

    centers = [
        [0, 0, 0],
        [1 / 3, 1 / 3, 1 / 3],
        [2 / 3, 2 / 3, 2 / 3],
    ]

    soap_arguments = {
        "n_max": 2,
        "l_max": 2,
        "r_cut": 2.0,
        "sigma": 0.55,
        "species": ["H", "C"],
        "crossover": True,
    }
    return [system, centers, soap_arguments]


def get_soap_gto_l_max_setup():
    """Returns an atomic system and SOAP parameters which are ideal for quickly
    testing the correctness of SOAP values with large l_max and the GTO basis.
    Minimizing the computation is important because the calculating the
    numerical benchmark values is very expensive.
    """
    # We need a lot of atoms in order to make the values at large l-values to
    # be above numerical precision and in order to have good distribution for
    # all l-components.
    x, y, z = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2))
    positions = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(float)
    rng = np.random.RandomState(42)
    positions += rng.random((27, 3)) - 0.5

    # One atom should be exactly at origin because it is an exceptional
    # location that needs to be tested.
    positions[13, :] = [0, 0, 0]

    system = Atoms(
        positions=positions,
        symbols=len(positions) * ["H"],
    )

    centers = [[0, 0, 0]]

    # Making sigma small enough ensures that the smaller l-components are not
    # screened by big fluffy gaussians.
    soap_arguments = {
        "n_max": 1,
        "l_max": 20,
        "r_cut": 2.0,
        "sigma": 0.1,
        "species": ["H"],
        "crossover": False,
    }
    return (system, centers, soap_arguments)


def get_soap_polynomial_l_max_setup():
    """Returns an atomic system and SOAP parameters which are ideal for quickly
    testing the correctness of SOAP values with large l_max and the polynomial
    basis.  Minimizing the computation is important because the calculating the
    numerical benchmark values is very expensive.
    """
    # We need a lot of atoms in order to make the values at large l-values to
    # be above numerical precision and in order to have good distribution for
    # all l-components.
    x, y, z = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), np.arange(-1, 2))
    positions = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(float)
    rng = np.random.RandomState(42)
    positions += rng.random((27, 3)) - 0.5

    # One atom should be exactly at origin because it is an exceptional
    # location that needs to be tested.
    positions[13, :] = [0, 0, 0]

    system = Atoms(
        positions=positions,
        symbols=len(positions) * ["H"],
    )

    centers = [[0, 0, 0]]

    # Making sigma small enough ensures that the smaller l-components are not
    # screened by big fluffy gaussians.
    soap_arguments = {
        "n_max": 1,
        "l_max": 9,
        "r_cut": 2.0,
        "sigma": 0.1,
        "species": ["H"],
        "crossover": False,
    }
    return (system, centers, soap_arguments)


def get_ewald_sum_matrix_default_setup():
    """Returns an atomic system and Ewald sum matrix parameters for testing."""
    system = get_simple_finite()
    ewald_arguments = {"n_atoms_max": 3, "permutation": "none"}
    create_arguments = {
        "a": 0.5,
        "r_cut": 30,
        "g_cut": 20,
        "accuracy": None,
    }
    return (system, ewald_arguments, create_arguments)


def get_ewald_sum_matrix_automatic_setup():
    """Returns an atomic system and Ewald sum matrix parameters using accuracy
    to determine parameters.
    """
    system = get_simple_finite()
    ewald_arguments = {"n_atoms_max": 3, "permutation": "none"}
    create_arguments = {
        "a": None,
        "r_cut": None,
        "g_cut": None,
        "accuracy": 1e-6,
    }
    return (system, ewald_arguments, create_arguments)


def get_weights(r, weighting):
    """Calculates the weights given an array of radials distances and the
    weighting function setup.
    """
    fname = weighting.get("function")
    w0 = weighting.get("w0")
    n = r.shape[0]

    # No weighting specified
    if fname is None and w0 is None:
        return np.ones(n)
    else:
        # No weighting function, only w0
        if fname is None and w0 is not None:
            weights = np.ones(n)
            weights[r == 0] = w0
            return weights
        else:
            if fname == "poly":
                r0 = weighting["r0"]
                c = weighting["c"]
                m = weighting["m"]

                def f(r):
                    w = c * np.power(1 + 2 * (r / r0) ** 3 - 3 * (r / r0) ** 2, m)
                    w[r > r0] = 0
                    return w

                func = f
            elif fname == "pow":
                r0 = weighting["r0"]
                c = weighting["c"]
                d = weighting["d"]
                m = weighting["m"]
                func = lambda r: c / (d + np.power(r / r0, m))
            elif fname == "exp":
                r0 = weighting["r0"]
                c = weighting["c"]
                d = weighting["d"]
                func = lambda r: c / (d + np.exp(-r / r0))

            # Weighting function and w0
            weights = func(r)
            if w0 is not None:
                weights[r == 0] = w0
            return weights


def coefficients_gto(system, centers, args):
    """Used to numerically calculate the inner product coefficients of SOAP
    with GTO radial basis.
    """
    n_max = args["n_max"]
    l_max = args["l_max"]
    r_cut = args["r_cut"]
    sigma = args["sigma"]
    weighting = args.get("weighting")

    positions = system.get_positions()
    symbols = system.get_chemical_symbols()
    atomic_numbers = system.get_atomic_numbers()
    species_ordered = sorted(list(set(atomic_numbers)))
    n_elems = len(species_ordered)

    # Calculate the weights and decays of the radial basis functions.
    soap = SOAP(**args)
    soap.create(system, centers=centers)
    alphas = np.reshape(soap._alphas, [l_max + 1, n_max])
    betas = np.reshape(soap._betas, [l_max + 1, n_max, n_max])

    def rbf_gto(r, n, l):
        i_alpha = alphas[l, 0:n_max]
        i_beta = betas[l, n, 0:n_max]
        return (i_beta * r**l * np.exp(-i_alpha * r**2)).sum()

    return soap_integration(system, centers, args, rbf_gto)


def coefficients_polynomial(system, centers, args):
    """Used to numerically calculate the inner product coeffientes of SOAP
    with polynomial radial basis.
    """
    n_max = args["n_max"]
    r_cut = args["r_cut"]
    atomic_numbers = system.get_atomic_numbers()

    # Calculate the overlap of the different polynomial functions in a
    # matrix S. These overlaps defined through the dot product over the
    # radial coordinate are analytically calculable: Integrate[(rc - r)^(a
    # + 2) (rc - r)^(b + 2) r^2, {r, 0, rc}]. Then the weights B that make
    # the basis orthonormal are given by B=S^{-1/2}
    S = np.zeros((n_max, n_max))
    for i in range(1, n_max + 1):
        for j in range(1, n_max + 1):
            S[i - 1, j - 1] = (2 * (r_cut) ** (7 + i + j)) / (
                (5 + i + j) * (6 + i + j) * (7 + i + j)
            )
    betas = sqrtm(np.linalg.inv(S))

    def rbf_polynomial(r, n, l):
        poly = 0
        for k in range(1, n_max + 1):
            poly += betas[n, k - 1] * (r_cut - np.clip(r, 0, r_cut)) ** (k + 2)
        return poly

    return soap_integration(system, centers, args, rbf_polynomial)


def soap_integration(system, centers, args, rbf_function):
    """Used to numerically calculate the inner product coeffientes of SOAP
    with polynomial radial basis.
    """
    n_max = args["n_max"]
    l_max = args["l_max"]

    positions = system.get_positions()
    atomic_numbers = system.get_atomic_numbers()
    species_ordered = sorted(list(set(atomic_numbers)))
    n_elems = len(species_ordered)

    p_args = []
    p_index = []
    for i, ipos in enumerate(centers):
        for iZ, Z in enumerate(species_ordered):
            indices = np.argwhere(atomic_numbers == Z).flatten()
            elem_pos = positions[indices]
            # This centers the coordinate system at the soap center
            elem_pos -= ipos
            for n in range(n_max):
                for l in range(l_max + 1):
                    for im, m in enumerate(range(-l, l + 1)):
                        p_args.append((args, n, l, m, elem_pos, rbf_function))
                        p_index.append((i, iZ, n, l, im))

    results = Parallel(n_jobs=8, verbose=1)(delayed(integral)(*a) for a in p_args)

    coeffs = np.zeros((len(centers), n_elems, n_max, l_max + 1, 2 * l_max + 1))
    for index, value in zip(p_index, results):
        coeffs[index] = value
    return coeffs


def integral(args, n, l, m, elem_pos, rbf_function):
    r_cut = args["r_cut"]
    sigma = args["sigma"]
    weighting = args.get("weighting")

    # Integration limits for radius
    r1 = 0.0
    r2 = r_cut + 5

    # Integration limits for theta
    t1 = 0
    t2 = np.pi

    # Integration limits for phi
    p1 = 0
    p2 = 2 * np.pi

    def soap_coeff(phi, theta, r):
        # Regular spherical harmonic, notice the abs(m)
        # needed for constructing the real form
        ylm_comp = scipy.special.sph_harm(
            np.abs(m), l, phi, theta
        )  # NOTE: scipy swaps phi and theta

        # Construct real (tesseral) spherical harmonics for
        # easier integration without having to worry about
        # the imaginary part. The real spherical harmonics
        # span the same space, but are just computationally
        # easier.
        ylm_real = np.real(ylm_comp)
        ylm_imag = np.imag(ylm_comp)
        if m < 0:
            ylm = np.sqrt(2) * (-1) ** m * ylm_imag
        elif m == 0:
            ylm = ylm_comp
        else:
            ylm = np.sqrt(2) * (-1) ** m * ylm_real

        # Atomic density
        rho = 0
        ix = elem_pos[:, 0]
        iy = elem_pos[:, 1]
        iz = elem_pos[:, 2]
        ri_squared = ix**2 + iy**2 + iz**2
        rho = np.exp(
            -1
            / (2 * sigma**2)
            * (
                r**2
                + ri_squared
                - 2
                * r
                * (
                    np.sin(theta) * np.cos(phi) * ix
                    + np.sin(theta) * np.sin(phi) * iy
                    + np.cos(theta) * iz
                )
            )
        )
        if weighting:
            weights = get_weights(np.sqrt(ri_squared), weighting)
            rho *= weights
        rho = rho.sum()

        # Jacobian
        jacobian = np.sin(theta) * r**2

        return rbf_function(r, n, l) * ylm * rho * jacobian

    cnlm = tplquad(
        soap_coeff,
        r1,
        r2,
        lambda r: t1,
        lambda r: t2,
        lambda r, theta: p1,
        lambda r, theta: p2,
        epsabs=1e-6,
        epsrel=1e-4,
    )
    integral, error = cnlm

    return integral


def save_gto_coefficients():
    """Used to precalculate the numerical SOAP coefficients for a test
    system. Calculating these takes a significant amount of time, so during
    tests these preloaded values are used.
    """
    system, centers, args = get_soap_gto_l_max_setup()
    coeffs = coefficients_gto(system, centers, args)
    np.save(
        folder / "gto_coefficients_{n_max}_{l_max}_{r_cut}_{sigma}.npy".format(**args),
        coeffs,
    )


def save_poly_coefficients():
    """Used to precalculate the numerical SOAP coefficients for a test
    system. Calculating these takes a significant amount of time, so during
    tests these preloaded values are used.
    """
    system, centers, args = get_soap_polynomial_l_max_setup()
    coeffs = coefficients_polynomial(system, centers, args)
    np.save(
        folder
        / "polynomial_coefficients_{n_max}_{l_max}_{r_cut}_{sigma}.npy".format(**args),
        coeffs,
    )


def calculate_ewald(system, a=None, r_cut=None, g_cut=None, accuracy=None):
    """Used to precalculate the Ewald summation results using pymatgen."""
    positions = system.get_positions()
    atomic_num = system.get_atomic_numbers()
    n_atoms = len(system)
    energy = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                pos = [positions[i]]
                sym = [atomic_num[i]]
            else:
                pos = [positions[i], positions[j]]
                sym = [atomic_num[i], atomic_num[j]]

            i_sys = Atoms(
                cell=system.get_cell(),
                positions=pos,
                symbols=sym,
                pbc=True,
            )

            structure = Structure(
                lattice=i_sys.get_cell(),
                species=i_sys.get_atomic_numbers(),
                coords=i_sys.get_scaled_positions(),
            )
            structure.add_oxidation_state_by_site(i_sys.get_atomic_numbers())
            ewald = EwaldSummation(
                structure,
                eta=a,
                real_space_cut=r_cut,
                recip_space_cut=g_cut,
                acc_factor=-np.log(accuracy) if accuracy else 12.0,
            )
            energy[i, j] = ewald.total_energy
    return energy


def save_ewald(system, args):
    coeffs = calculate_ewald(system, **args)
    np.save(folder / "ewald_{a}_{r_cut}_{g_cut}_{accuracy}.npy".format(**args), coeffs)


def load_ewald(args):
    return np.load(folder / "ewald_{a}_{r_cut}_{g_cut}_{accuracy}.npy".format(**args))


if __name__ == "__main__":
    # save_gto_coefficients()
    # save_poly_coefficients()
    system, _, create_args = get_ewald_sum_matrix_default_setup()
    save_ewald(system, create_args)
    system, _, create_args = get_ewald_sum_matrix_automatic_setup()
    save_ewald(system, create_args)
