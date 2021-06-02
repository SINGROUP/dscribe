import numpy as np
import scipy
from scipy.integrate import tplquad
from scipy.linalg import sqrtm
from ase import Atoms
from dscribe.descriptors import SOAP


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
        symbols=["H", "C"]
    )

    centers = [
        [0, 0, 0],
        [1 / 3, 1 / 3, 1 / 3],
        [2 / 3, 2 / 3, 2 / 3],
    ]

    soap_arguments = {
        "nmax": 2,
        "lmax": 2,
        "rcut": 2.0,
        "sigma": 0.55,
        "species": ["H", "C"],
        "crossover": True
    }
    return [system, centers, soap_arguments]


def get_soap_lmax_setup():
    """Returns an atomic system and SOAP parameters which are ideal for quickly
    testing the correctness of SOAP values with large lmax. Minimizing the
    computation is important because the calculating the numerical benchmark
    values is very expensive.
    """
    # Calculate the numerical power spectrum
    system = Atoms(
        positions=[
            [0.0, 0.0, 0.0],
            [-0.3, 0.5, 0.4],
            # [0.2, 0.2, 0.2],
            # # [-0.2, -0.2, -0.2],
        ],
        symbols=["H", "C"]
        # # symbols=["H", "H", "H", "H"]
    )

    centers = [[0, 0, 0]]

    soap_arguments = {
        "nmax": 2,
        "lmax": 2,
        "rcut": 2.0,
        "sigma": 0.35,
        "species": ["H", "C"],
        "crossover": True
    }
    return [system, centers, soap_arguments]


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
    nmax = args["nmax"]
    lmax = args["lmax"]
    rcut = args["rcut"]
    sigma = args["sigma"]
    weighting = args.get("weighting")

    # Calculate the analytical power spectrum and the weights and decays of
    # the radial basis functions.
    soap = SOAP(**args)
    soap.create(system, positions=centers)
    alphas = np.reshape(soap._alphas, [lmax + 1, nmax])
    betas = np.reshape(soap._betas, [lmax + 1, nmax, nmax])

    positions = system.get_positions()
    symbols = system.get_chemical_symbols()
    species = system.get_atomic_numbers()
    elements = set(system.get_atomic_numbers())
    n_elems = len(elements)

    # Integration limits for radius
    r1 = 0.0
    r2 = rcut + 5

    # Integration limits for theta
    t1 = 0
    t2 = np.pi

    # Integration limits for phi
    p1 = 0
    p2 = 2 * np.pi

    coeffs = np.zeros((len(centers), n_elems, nmax, lmax + 1, 2 * lmax + 1))
    for i, ipos in enumerate(centers):
        for iZ, Z in enumerate(elements):
            indices = np.argwhere(species == Z)[0]
            elem_pos = positions[indices]
            # This centers the coordinate system at the soap center
            elem_pos -= ipos
            for n in range(nmax):
                for l in range(lmax + 1):
                    for im, m in enumerate(range(-l, l + 1)):

                        # Calculate numerical coefficients
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

                            # Spherical gaussian type orbital
                            i_alpha = alphas[l, 0:nmax]
                            i_beta = betas[l, n, 0:nmax]
                            gto = (
                                i_beta * r ** l * np.exp(-i_alpha * r ** 2)
                            ).sum()

                            # Atomic density
                            rho = 0
                            ix = elem_pos[:, 0]
                            iy = elem_pos[:, 1]
                            iz = elem_pos[:, 2]
                            ri_squared = ix ** 2 + iy ** 2 + iz ** 2
                            rho = np.exp(
                                -1
                                / (2 * sigma ** 2)
                                * (
                                    r ** 2
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
                                weights = get_weights(
                                    np.sqrt(ri_squared), weighting
                                )
                                rho *= weights
                            rho = rho.sum()

                            # Jacobian
                            jacobian = np.sin(theta) * r ** 2

                            return gto * ylm * rho * jacobian

                        cnlm = tplquad(
                            soap_coeff,
                            r1,
                            r2,
                            lambda r: t1,
                            lambda r: t2,
                            lambda r, theta: p1,
                            lambda r, theta: p2,
                            epsabs=0.001,
                            epsrel=0.001,
                        )
                        integral, error = cnlm
                        coeffs[i, iZ, n, l, im] = integral

    return coeffs


def coefficients_polynomial(system, centers, args):
    """Used to numerically calculate the inner product coeffientes of SOAP
    with polynomial radial basis.
    """
    nmax = args["nmax"]
    lmax = args["lmax"]
    rcut = args["rcut"]
    sigma = args["sigma"]
    weighting = args.get("weighting")

    positions = system.get_positions()
    symbols = system.get_chemical_symbols()
    species = system.get_atomic_numbers()
    elements = set(system.get_atomic_numbers())
    n_elems = len(elements)

    # Integration limits for radius
    r1 = 0.0
    r2 = rcut + 5

    # Integration limits for theta
    t1 = 0
    t2 = np.pi

    # Integration limits for phi
    p1 = 0
    p2 = 2 * np.pi

    # Calculate the overlap of the different polynomial functions in a
    # matrix S. These overlaps defined through the dot product over the
    # radial coordinate are analytically calculable: Integrate[(rc - r)^(a
    # + 2) (rc - r)^(b + 2) r^2, {r, 0, rc}]. Then the weights B that make
    # the basis orthonormal are given by B=S^{-1/2}
    S = np.zeros((nmax, nmax))
    for i in range(1, nmax + 1):
        for j in range(1, nmax + 1):
            S[i - 1, j - 1] = (2 * (rcut) ** (7 + i + j)) / (
                (5 + i + j) * (6 + i + j) * (7 + i + j)
            )
    betas = sqrtm(np.linalg.inv(S))

    coeffs = np.zeros((len(centers), n_elems, nmax, lmax + 1, 2 * lmax + 1))
    for i, ipos in enumerate(centers):
        for iZ, Z in enumerate(elements):
            indices = np.argwhere(species == Z)[0]
            elem_pos = positions[indices]
            # This centers the coordinate system at the soap center
            elem_pos -= ipos
            for n in range(nmax):
                for l in range(lmax + 1):
                    for im, m in enumerate(range(-l, l + 1)):

                        # Calculate numerical coefficients
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

                            # Polynomial basis
                            poly = 0
                            for k in range(1, nmax + 1):
                                poly += betas[n, k - 1] * (
                                    rcut - np.clip(r, 0, rcut)
                                ) ** (k + 2)

                            # Atomic density
                            rho = 0
                            ix = elem_pos[:, 0]
                            iy = elem_pos[:, 1]
                            iz = elem_pos[:, 2]
                            ri_squared = ix ** 2 + iy ** 2 + iz ** 2
                            rho = np.exp(
                                -1
                                / (2 * sigma ** 2)
                                * (
                                    r ** 2
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
                                weights = get_weights(
                                    np.sqrt(ri_squared), weighting
                                )
                                rho *= weights
                            rho = rho.sum()

                            # Jacobian
                            jacobian = np.sin(theta) * r ** 2

                            return poly * ylm * rho * jacobian

                        cnlm = tplquad(
                            soap_coeff,
                            r1,
                            r2,
                            lambda r: t1,
                            lambda r: t2,
                            lambda r, theta: p1,
                            lambda r, theta: p2,
                            epsabs=0.0001,
                            epsrel=0.0001,
                        )
                        integral, error = cnlm
                        coeffs[i, iZ, n, l, im] = integral

    return coeffs


def save_gto_coefficients():
    """Used to precalculate the numerical SOAP coefficients for a test
    system. Calculating these takes a significant amount of time, so during
    tests these preloaded values are used.
    """
    system, centers, args = get_soap_lmax_setup()
    coeffs = coefficients_gto(system, centers, args)
    np.save(
        "gto_coefficients_{nmax}_{lmax}_{rcut}_{sigma}.npy".format(**args),
        coeffs,
    )


def save_poly_coefficients():
    """Used to precalculate the numerical SOAP coefficients for a test
    system. Calculating these takes a significant amount of time, so during
    tests these preloaded values are used.
    """
    system, centers, args = get_soap_lmax_setup()
    coeffs = coefficients_polynomial(system, centers, args)
    np.save(
        "polynomial_coefficients_{nmax}_{lmax}_{rcut}_{sigma}.npy".format(**args),
        coeffs,
    )


def load_gto_coefficients(args):
    return np.load("gto_coefficients_{nmax}_{lmax}_{rcut}_{sigma}.npy".format(**args))


def load_polynomial_coefficients(args):
    return np.load("polynomial_coefficients_{nmax}_{lmax}_{rcut}_{sigma}.npy".format(**args))


# if __name__ == "__main__":
    # save_gto_coefficients()
    # save_poly_coefficients()
