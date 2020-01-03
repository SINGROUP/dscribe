from time import time
import numpy as np
from tqdm import tqdm
from ase import Atoms
from dscribe.descriptors import SOAP
import matplotlib.pyplot as mpl
import pkg_resources

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['cm']
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['legend.fontsize'] = 18


def test_soap(version):
    """Tests how the SOAP descriptor calculation scales with system size.
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
        # Loop over different system sizes
        for ncells in tqdm(range(5, 20)):

            natoms = 2 * ncells ** 3
            soap_generator = SOAP(rcut=3.0, nmax=nmax, lmax=lmax, species=["Ni", "Ti"], rbf=rbf, crossover=True, periodic=True)

            a = 2.993
            niti = Atoms(
                "NiTi",
                positions=[[0.0, 0.0, 0.0], [a / 2, a / 2, a / 2]],
                cell=[a, a, a],
                pbc=[1, 1, 1],
            )

            # Replicate system
            niti = niti * ncells
            a *= ncells

            t0 = time()
            soap_generator.create(niti)
            t1 = time()

            N.append(natoms)
            t.append(t1 - t0)

        N = np.array(N)
        t = np.array(t)

        ax.plot(N, t, "o--", label="{}".format(rbf))

    mpl.legend()
    mpl.savefig("soap_scaling_{}.pdf".format(version))

version = pkg_resources.get_distribution('dscribe').version
test_soap(version)
