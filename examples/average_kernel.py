"""Demonstrates the use of the utility AverageKernel
In this example global structures are compared based on
averaging the local descriptors of each atom within
the structures"""
from dscribe.descriptors import SOAP, ACSF
import ase, pickle
import numpy as np
from dscribe.utils import AverageKernel
from ase.build import molecule
from ase.collections import g2
import time

# Choose descriptor
descriptor = "SOAP"

# Compute local descriptors
all_atomtypes = [1,6]
#all_atomtypes = []
if descriptor == "SOAP":
    desc = SOAP(all_atomtypes,
        8.0, 2, 0, periodic=False,crossover=True)
    print(desc.get_number_of_features())
elif descriptor == "ACSF":
    desc = ACSF(n_atoms_max=15, types=[1,6,7,8],bond_params=[[1,2,], [4,5,]],
        bond_cos_params=[1,2,3,4],
        ang4_params=[[1,2,3],[3,1,4], [4,5,6], [7,8,9]],
        ang5_params=[[1,2,3],[3,1,4], [4,5,6], [7,8,9]], flatten=False)
else:
    print("Add your local descriptor here")
    exit(0)

ave = AverageKernel()
desc_list = []
atomic_numbers_list = []
ase_atoms_list = []
all_atomtypes = [1,6,]
# choose a few molecules from ase database
for molname in g2.names:
    atoms = molecule(molname)
    atomic_numbers = atoms.get_atomic_numbers()
    sorted_set_atomic_numbers = sorted(set(atomic_numbers))
    leftover = set(atomic_numbers) - set(all_atomtypes)
    if len(leftover) > 0:
        continue
    leftover = set(all_atomtypes) - set(atomic_numbers)
    if len(leftover) > 0:
        continue
    local_a = desc.create(atoms)
    if len(local_a) == 0:
        continue
    else:
        for i in range(1):
            desc_list.append(local_a)
            atomic_numbers_list.append(sorted_set_atomic_numbers)
            ase_atoms_list.append(atoms)

# Compute the global distance matrix
print(len(desc_list))
print(atomic_numbers_list)
distance_matrix = ave.get_global_distance_matrix(desc_list, metric = 'euclidean')
print(distance_matrix[0:5,0:5])
print(distance_matrix.shape)



###  Example for circumventing large descriptor arrays ###
def get_distance_from_heterogeneous_soap(average_a, average_b, atomic_numbers_a, atomic_numbers_b, chunksize):
    overlap_ab, overlap_ba, noverlap_a, noverlap_b = _get_overlap_ids(atomic_numbers_a, atomic_numbers_b, chunksize)

    distance = _get_distance_from_chunks(average_a, average_b, overlap_ab, overlap_ba, noverlap_a, noverlap_b)

    return distance


def _get_distance_from_chunks(local_a, local_b, overlap_ab, overlap_ba, noverlap_a, noverlap_b ):
    diff = local_a[overlap_ab] - local_b[overlap_ba]
    only_a = local_a[noverlap_a]
    only_b = local_b[noverlap_b]
    combined = np.concatenate((diff, only_a, only_b))
    distance = np.linalg.norm(combined)
    return distance



def _get_overlap_ids(atomic_numbers_a, atomic_numbers_b, chunksize):
    len_a = len(atomic_numbers_a)
    len_b = len(atomic_numbers_b)
    chunkcounter = np.arange(chunksize, dtype=int).reshape(1,-1)
    idsm_a = np.zeros((len_a, len_a), dtype=int)
    idsm_a[np.diag_indices(len_a)] = range(len_a)
    idsm_a[np.triu_indices(len_a, k = 1)] = np.arange(len_a , len_a * (len_a + 1) / 2)
    idsm_b = np.zeros((len_b, len_b), dtype=int)
    idsm_b[np.diag_indices(len_b)] = range(len_b)
    idsm_b[np.triu_indices(len_b, k = 1)] = np.arange(len_b , len_b * (len_b + 1) / 2)
    is_overlap_a = np.isin(atomic_numbers_a, atomic_numbers_b)
    is_overlap_b = np.isin(atomic_numbers_b, atomic_numbers_a)
    noverlap_a   = np.isin(atomic_numbers_a, atomic_numbers_b, invert=True)
    noverlap_b   = np.isin(atomic_numbers_b, atomic_numbers_a, invert=True)
    chunk_ids_ab = idsm_a[np.ix_(is_overlap_a, is_overlap_a)]

    if chunk_ids_ab.shape[0] > 0:
        chunk_ids_ab = chunk_ids_ab[np.triu_indices(chunk_ids_ab.shape[0], k = 0)]
    chunk_ids_ba = idsm_b[np.ix_(is_overlap_b, is_overlap_b)]
    if chunk_ids_ba.shape[0] > 0:
        chunk_ids_ba = chunk_ids_ba[np.triu_indices(chunk_ids_ba.shape[0], k = 0)]
    chunk_ids_a  = idsm_a[np.ix_(noverlap_a, noverlap_a)]
    if chunk_ids_a.shape[0] > 0:
        chunk_ids_a = chunk_ids_a[np.triu_indices(chunk_ids_a.shape[0], k = 0)]
    chunk_ids_b  = idsm_b[np.ix_(noverlap_b, noverlap_b)]
    if chunk_ids_b.shape[0] > 0:
        chunk_ids_b = chunk_ids_b[np.triu_indices(chunk_ids_b.shape[0], k = 0)]

    overlap_ab  = (np.multiply(chunk_ids_ab.reshape(-1,1), chunksize) + chunkcounter).flatten()
    overlap_ba  = (np.multiply(chunk_ids_ba.reshape(-1,1), chunksize) + chunkcounter).flatten()
    noverlap_a  = (np.multiply(chunk_ids_a.reshape(-1,1), chunksize) + chunkcounter).flatten()
    noverlap_b  = (np.multiply(chunk_ids_b.reshape(-1,1), chunksize) + chunkcounter).flatten()

    return overlap_ab, overlap_ba, noverlap_a, noverlap_b

def get_chunksize(nmax, lmax):
    chunksize = (lmax + 1) * nmax * (nmax +1) / 2
    return int(chunksize)

desc_list = []
atomic_numbers_list = []

rcut = 7.0
nmax = 10
lmax = 8

# load from pickle file
pickle_file = "atoms_10000.pickle"
with open(pickle_file, 'rb') as f:
    pickle_object = pickle.load(f)

#atomic_numbers = []
#for atoms in pickle_object:
#    atomic_numbers.extend(atoms.get_atomic_numbers())
#sorted_set_atomic_numbers = sorted(set(atomic_numbers))
#
#print("sorted_set_atomic_numbers")
#print(sorted_set_atomic_numbers)
#print(len(sorted_set_atomic_numbers))

for idx, atoms in enumerate(pickle_object):
#for idx, atoms in enumerate(ase_atoms_list):
    atomic_numbers = atoms.get_atomic_numbers()
    sorted_set_atomic_numbers = sorted(set(atomic_numbers))
    #desc = SOAP(atomic_numbers = sorted_set_atomic_numbers, rcut = rcut, nmax = nmax, lmax = lmax, periodic=False,crossover=True)
    desc = SOAP(atomic_numbers = sorted_set_atomic_numbers, rcut = rcut, nmax = nmax, lmax = lmax, periodic=True,crossover=True)
    local_a = desc.create(atoms)
    average_a = local_a.mean(axis = 0)

    desc_list.append(average_a)
    atomic_numbers_list.append(sorted_set_atomic_numbers)


chunksize = get_chunksize(nmax, lmax)
print("chunksize", chunksize)

N = len(desc_list)
ave_dist_matrix = np.zeros((N, N), dtype=np.float64)

t0 = time.time()
for row_id, average_a, atomic_numbers_a in zip(range(N), desc_list, atomic_numbers_list):
    for col_id, average_b, atomic_numbers_b in zip(range(N), desc_list, atomic_numbers_list):
        distance = get_distance_from_heterogeneous_soap(average_a, average_b, atomic_numbers_a, atomic_numbers_b, chunksize)
        if col_id > row_id:
            ave_dist_matrix[row_id, col_id] = distance
t1 = time.time()
print(ave_dist_matrix[0:5,0:5])

print(ave_dist_matrix.shape)
print("time for computing distances", t1 -t0)

np.save("soap_n" + str(nmax) + "l" + str(lmax) + "r" + str(rcut) + ".npy", ave_dist_matrix)

print("done saving")
