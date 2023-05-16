from dscribe.descriptors import SOAP

species = ["H", "C", "O", "N"]
r_cut = 6.0
n_max = 8
l_max = 6

# Setting up the SOAP descriptor
soap = SOAP(
    species=species,
    periodic=False,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
)

# Creation
from ase.build import molecule

# Molecule created as an ASE.Atoms
water = molecule("H2O")

# Create SOAP output for the system
soap_water = soap.create(water, centers=[0])

print(soap_water)
print(soap_water.shape)

# Create output for multiple system
samples = [molecule("H2O"), molecule("NO2"), molecule("CO2")]
centers = [[0], [1, 2], [1, 2]]
coulomb_matrices = soap.create(samples, centers)            # Serial
coulomb_matrices = soap.create(samples, centers, n_jobs=2)  # Parallel

# Lets change the SOAP setup and see how the number of features changes
small_soap = SOAP(species=species, r_cut=r_cut, n_max=2, l_max=0)
big_soap = SOAP(species=species, r_cut=r_cut, n_max=9, l_max=9)
n_feat1 = small_soap.get_number_of_features()
n_feat2 = big_soap.get_number_of_features()
print(n_feat1, n_feat2)

# Periodic systems
from ase.build import bulk

copper = bulk('Cu', 'fcc', a=3.6, cubic=True)
print(copper.get_pbc())
periodic_soap = SOAP(
    species=[29],
    r_cut=r_cut,
    n_max=n_max,
    l_max=n_max,
    periodic=True,
    sparse=False
)

soap_copper = periodic_soap.create(copper)

print(soap_copper)
print(soap_copper.sum(axis=1))

# Locations
# The locations of specific element combinations can be retrieved like this.
hh_loc = soap.get_location(("H", "H"))
ho_loc = soap.get_location(("H", "O"))

# These locations can be directly used to slice the corresponding part from an
# SOAP output for e.g. plotting.
soap_water[0, hh_loc]
soap_water[0, ho_loc]

# Sparse output
soap = SOAP(
    species=species,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    sparse=True
)
soap_water = soap.create(water)
print(type(soap_water))

soap = SOAP(
    species=species,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    sparse=False
)
soap_water = soap.create(water)
print(type(soap_water))

# Average output
average_soap = SOAP(
    species=species,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    average="inner",
    sparse=False
)

soap_water = average_soap.create(water)
print("Average SOAP water: ", soap_water.shape)

methanol = molecule('CH3OH')
soap_methanol = average_soap.create(methanol)
print("Average SOAP methanol: ", soap_methanol.shape)

h2o2 = molecule('H2O2')
soap_peroxide = average_soap.create(h2o2)
print("Average SOAP peroxide: ", soap_peroxide.shape)

# Distance
from scipy.spatial.distance import pdist, squareform
import numpy as np

molecules = np.vstack([soap_water, soap_methanol, soap_peroxide])
distance = squareform(pdist(molecules))
print("Distance matrix: water - methanol - peroxide: ")
print(distance)
