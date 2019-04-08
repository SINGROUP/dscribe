from dscribe.descriptors import SOAP

atomic_numbers = [1, 6, 7, 8]
rcut = 6.0
nmax = 8
lmax = 6

# Setting up the SOAP descriptor
soap = SOAP(
    species=atomic_numbers,
    periodic=False,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
)

# Creation
from ase.build import molecule

# Molecule created as an ASE.Atoms
water = molecule("H2O")

# Create SOAP output for the system
soap_water = soap.create(water, positions=[0])

print(soap_water)
print(soap_water.shape)

# Create output for multiple system
samples = [molecule("H2O"), molecule("NO2"), molecule("CO2")]
positions = [[0], [1, 2], [1, 2]]
coulomb_matrices = soap.create(samples, positions)            # Serial
coulomb_matrices = soap.create(samples, positions, n_jobs=2)  # Parallel

# Lets change the SOAP setup and see how features change
minimal_soap = SOAP(species=atomic_numbers, rcut=rcut, nmax=2, lmax=0)
n_features = minimal_soap.get_number_of_features()

print("minimal number of features", n_features)

# Lets try SOAP on a new molecule
print("previous sum", soap_water.sum(axis=1))

atomic_numbers = [1, 6, 8]
soap = SOAP(species=atomic_numbers, rcut=rcut, nmax=nmax, lmax=lmax)

soap_water = soap.create(water, positions=[0])

print(soap_water.shape)
print("unchanged sum", soap_water.sum(axis=1))

methanol = molecule("CH3OH")
soap_methanol = soap.create(methanol, positions=[0])

print(soap_methanol.shape)

# Periodic systems
from ase.build import bulk

copper = bulk('Cu', 'fcc', a=3.6, cubic=True)
print(copper.get_pbc())
periodic_soap = SOAP(
    species=[29],
    rcut=rcut,
    nmax=nmax,
    lmax=nmax,
    periodic=True,
    sparse=False
)

soap_copper = periodic_soap.create(copper)

print(soap_copper)
print(soap_copper.sum(axis=1))

# Sparse output
soap = SOAP(
    species=atomic_numbers,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    sparse=True
)
soap_water = soap.create(water)
print(type(soap_water))

soap = SOAP(
    species=atomic_numbers,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    sparse=False
)
soap_water = soap.create(water)
print(type(soap_water))

# Average output
average_soap = SOAP(
    species=atomic_numbers,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
    average=True,
    sparse=False
)

soap_water = average_soap.create(water)
print("average soap water", soap_water.shape)

soap_methanol = average_soap.create(methanol)
print("average soap methanol", soap_methanol.shape)

h2o2 = molecule('H2O2')
soap_peroxide = average_soap.create(h2o2)

# Distance
from scipy.spatial.distance import pdist, squareform
import numpy as np

molecules = np.vstack([soap_water, soap_methanol, soap_peroxide])
distance = squareform(pdist(molecules))
print("distance matrix: water - methanol - H2O2")
print(distance)
