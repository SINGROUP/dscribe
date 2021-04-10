from time import time
import numpy as np
from tqdm import tqdm
from ase import Atoms
from dscribe.descriptors import SOAP
import matplotlib.pyplot as mpl
import pkg_resources

mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 18

# Periodic test system.
a = 2.993
system_periodic = Atoms(
    "NiTi",
    positions=[[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]],
    cell=[a, a, a],
    pbc=True,
)


def soap_gto_vs_polynomial(version):
    """GTO vs polynomial RBF scaling.
    """
    nmax = 4
    lmax = 4
    fig = mpl.figure(figsize=[9, 7])
    ax = fig.add_subplot(111)
    ax.set_title("SOAP nmax={}, lmax={}, version={}".format(nmax, lmax, version))
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Time (s)")

    for rbf in ["gto", "polynomial"]:
        N = []
        t = []
        for ncells in tqdm(range(5, 15)):
            soap_generator = SOAP(rcut=3.0, nmax=nmax, lmax=lmax, species=["Ni", "Ti"], rbf=rbf, crossover=True, periodic=True)
            i_system = system_periodic.copy() * ncells
            t0 = time()
            soap_generator.create(i_system)
            t1 = time()
            N.append(len(i_system))
            t.append(t1 - t0)

        ax.plot(N, t, "o--", label="{}".format(rbf))

    mpl.legend()
    mpl.show()


def soap_derivatives(version):
    """Tests how the SOAP derivative calculation (numerical+analytical) scales
    with system size.
    """
    nmax = 4
    lmax = 4
    fig = mpl.figure(figsize=[9, 7])
    ax = fig.add_subplot(111)
    ax.set_title("SOAP derivatives nmax={}, lmax={}, version={}".format(nmax, lmax, version))
    ax.set_xlabel("lmax")
    ax.set_ylabel("Time (s)")
    system = system_periodic*(5,5,5)

    for method in ["numerical", "analytical"]:
        N = []
        t = []
        for ncells in tqdm(range(1, 5)):
            i_system = system.copy() * [ncells, 1, 1]
            soap_generator = SOAP(
                rcut=3.0,
                nmax=nmax,
                lmax=lmax,
                species=["Ni", "Ti"],
                rbf="gto",
                crossover=True,
                periodic=False
            )

            t0 = time()
            der, des = soap_generator.derivatives(i_system, method=method)
            t1 = time()
            N.append(len(i_system))
            t.append(t1 - t0)

        ax.plot(N, t, "o--", label="{}".format(method))

    mpl.legend()
    mpl.show()


def soap_cartesian_vs_imaginary(version):
    """Tests the performance of cartesian SOAP GTO vs. imaginary SOAP GTO.
    """
    nmax = 6
    lmax = 6
    fig = mpl.figure(figsize=[9, 7])
    ax = fig.add_subplot(111)
    ax.set_title("SOAP nmax={}, lmax={}, version={}".format(nmax, lmax, version))
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Time (s)")
    system = system_periodic*(5,5,5)

    for method in ["imaginary", "tesseral"]:

        N = []
        t = []
        for ncells in tqdm(range(9, 10)):
            soap_generator = SOAP(
                rcut=3.0,
                nmax=nmax,
                lmax=lmax,
                species=["Ni", "Ti"],
                rbf="gto",
                crossover=True,
                periodic=False
            )

            i_system = system.copy() * [ncells, 1, 1]
            t0 = time()
            if method == "imaginary":
                des = soap_generator.create_single(i_system)
            elif method == "tesseral":
                des = soap_generator._cartesian(i_system)
            else:
                raise
            t1 = time()
            N.append(len(i_system))
            t.append(t1 - t0)

        # ax.plot(N, t, "o--", label="{}".format(method))

    # mpl.legend()
    # mpl.show()


def soap_sparse_vs_dense(version):
    """Tests sparse vs. dense derivatives calculation.
    """
    nmax = 4
    lmax = 4
    fig = mpl.figure(figsize=[9, 7])
    ax = fig.add_subplot(111)
    ax.set_title("SOAP derivatives nmax={}, lmax={}, version={}".format(nmax, lmax, version))
    ax.set_xlabel("Number of atoms")
    ax.set_ylabel("Time (s)")
    system = system_periodic*(5,5,5)

    # Loop over different system sizes
    vals = []
    for sparse in [True, False]:
        N = []
        t = []
        for ncells in tqdm(range(1, 6)):
            soap_generator = SOAP(
                rcut=3.0,
                nmax=nmax,
                lmax=lmax,
                species=["Ni", "Ti"],
                rbf="gto",
                crossover=True,
                periodic=False,
                sparse=sparse,
            )

            i_system = system.copy() * [ncells, 1, 1]
            t0 = time()
            der, des = soap_generator.derivatives(i_system, method="analytical")
            vals.append(der)
            t1 = time()
            N.append(len(i_system))
            t.append(t1 - t0)
        ax.plot(N, t, "o--", label="sparse={}".format(sparse))

    mpl.legend()
    mpl.show()

version = pkg_resources.get_distribution('dscribe').version
# soap_gto_vs_polynomial(version)
# soap_derivatives(version)
soap_cartesian_vs_imaginary(version)
# soap_sparse_vs_dense(version)

