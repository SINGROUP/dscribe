import ase
from dscribe.descriptors import SOAP

#dscribe.descriptors.soap.SOAP(atomic_numbers, rcut, nmax, lmax, periodic=False, crossover=True, average=False, normalize=False, sparse=True)

# Instantiate #
###############
atomic_numbers = [1,8]
rcut = 6.0
nmax = 8
lmax = 6

soap = SOAP(atomic_numbers, rcut, nmax, lmax)

# Create #
##########

from ase.structure import molecule

water = molecule("H2O")

soap_water = soap.create(water)

print(soap_water)
print(soap_water.shape)
print(soap_water.sum(axis = 1))


# adding another specie 
atomic_numbers = [1, 6, 8]
soap = SOAP(atomic_numbers, rcut, nmax, lmax)

soap_water = soap.create(water)

print(soap_water.shape)
print(soap_water.sum(axis = 1))


# other molecule
soap = SOAP(atomic_numbers, rcut, nmax, lmax)

methanol = molecule("CH3OH")
soap_methanol = soap.create(methanol)

print(soap_methanol.shape)


# changing nmax and lmax
minimal_soap = SOAP(atomic_numbers, rcut, 2, 0)

print("minimal number of features", minimal_soap.get_number_of_features())

minimal_soap_methanol = minimal_soap.create(methanol)

print(minimal_soap_methanol.shape)

# Optional Parameters #
#######################

# sparse

soap = SOAP(atomic_numbers, rcut, nmax, lmax, sparse = True)
soap_water = soap.create(water)
print(type(soap_water))

soap = SOAP(atomic_numbers, rcut, nmax, lmax, sparse = False)
soap_water = soap.create(water)
print(type(soap_water))


# average
average_soap = SOAP(atomic_numbers, rcut, nmax, lmax, average = True, sparse = False)

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


# periodic
from ase.build import bulk
copper = bulk('Cu', 'fcc', a=3.6, cubic = True)
print(copper.get_pbc())
atomic_numbers = [29]
periodic_soap = SOAP(atomic_numbers, rcut, nmax, lmax, periodic = True, sparse = False)

soap_copper = periodic_soap.create(copper)

print(soap_copper)
print(soap_copper.sum(axis = 1))




# Utilities #
#############

# create in batch
from dscribe.utils import batch_create

molecule_lst = [water, methanol]
batch = batch_create(average_soap, molecule_lst, n_proc = 1)

print(batch.shape)

# average kernel

from dscribe.utils import AverageKernel

ave = AverageKernel()

avemat = ave.get_global_distance_matrix([soap_water, soap_methanol])

print(avemat)


# rematch kernel

from dscribe.utils import RematchKernel
rematch = RematchKernel()

envkernels = rematch.get_all_envkernels([soap_water, soap_methanol])
remat = rematch.get_global_kernel(envkernels, gamma = 0.1, threshold = 1e-6)

print(remat)