from dscribe.descriptors import SOAP

atomic_numbers = [1, 8]
rcut = 6.0
nmax = 8
lmax = 6

# Setting up the SOAP descriptor
soap = SOAP(
    atomic_numbers=atomic_numbers,
    periodic=False,
    rcut=rcut,
    nmax=nmax,
    lmax=lmax,
)

# Creating an atomic system as an ase.Atoms-object
from ase.build import molecule

water = molecule("H2O")

# Create SOAP output for the system
soap_water = soap.create(water, positions=[0])

print(soap_water)
print(soap_water.shape)

# Lets change the SOAP setup and see how features change
minimal_soap = SOAP(atomic_numbers, rcut, 2, 0)
n_features = minimal_soap.get_number_of_features()

print("minimal number of features", n_features)

# Lets try SOAP on a new molecule
print("previous sum", soap_water.sum(axis=1))

atomic_numbers = [1, 6, 8]
soap = SOAP(atomic_numbers, rcut, nmax, lmax)

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
periodic_soap = SOAP([29], rcut, nmax, lmax, periodic=True, sparse=False)

soap_copper = periodic_soap.create(copper)

print(soap_copper)
print(soap_copper.sum(axis=1))

# Sparse output
soap = SOAP(atomic_numbers, rcut, nmax, lmax, sparse=True)
soap_water = soap.create(water)
print(type(soap_water))

soap = SOAP(atomic_numbers, rcut, nmax, lmax, sparse=False)
soap_water = soap.create(water)
print(type(soap_water))

# Average output
average_soap = SOAP(atomic_numbers, rcut, nmax, lmax, average=True, sparse=False)

soap_water = average_soap.create(water)
print("average soap water", soap_water.shape)

soap_methanol = average_soap.create(methanol)
print("average soap methanol", soap_methanol.shape)

h2o2 = molecule('H2O2')
soap_peroxide = average_soap.create(h2o2)

from scipy.spatial.distance import pdist, squareform
import numpy as np

molecules = np.vstack([soap_water, soap_methanol, soap_peroxide])
distance = squareform(pdist(molecules))
print("distance matrix: water - methanol - H2O2")
print(distance)

# Creating multiple descriptors in parallel
from dscribe.utils import batch_create

molecule_lst = [water, methanol]
batch = batch_create(average_soap, molecule_lst, n_proc=1)

print(batch.shape)
