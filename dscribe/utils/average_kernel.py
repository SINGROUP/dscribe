import dscribe
import numpy as np
import ase
from scipy.spatial.distance import cdist

class AverageKernel():
    """
    Rematch Kernel methods to go from local descriptors
    to global similarity
    """
    def __init__(self):
        return

    def compute_gaussian(self, ave_a,ave_b, gamma = 0.01):
        """Takes two matrices and computes the similarity
        based on the gaussian kernel.
        """
        return np.exp(-gamma * np.sqrt(((ave_a - ave_b)**2).sum()))

    def distance_matrix_to_gaussian_kernel(self, dist_matrix, gamma = 0.01):
        """Takes a distance matrix and computes the similarity
        based on the gaussian kernel.
        """
        return np.exp(-gamma * dist_matrix)

    def get_global_distance_matrix(self, desc_list, metric = 'euclidean'):
        """ Takes a list of M x N matrices, where M is the number of atoms
        in the system and N is the number of features of the descriptor.
        The matrices can be of different sizes.
        Returns a squared matrix with the size of the given dataset.
        Args:
            desc_list: list of NAxM matrices of structure A with NA atoms and M features.
            metric: Either string of valid cdist metric or custom function
        """
        #N = int(len(desc_list))
        ave_desc = self.average_descriptor(desc_list)

        dist_matrix = cdist(ave_desc, ave_desc, metric = metric)
        return dist_matrix


    def average_descriptor(self, desc_list):
        """
        Compute the average global similarity between two structures A and B,
        as reported in:
        Phys. Chem. Chem. Phys., 2016, 18, p. 13768
        Takes a list of NA x M matrices, where NA is the number of atoms
        in the system and M is the number of features of the descriptor.
        The matrices can be of different sizes.
        Returns a squared matrix with the size of the given dataset.
        Args:
            desc_list: list of NAxM matrices of structure A with NA atoms and M features.
        Returns:
            average_descriptor matrix: NxM matrix with N datapoints and M features
        """
        N = len(desc_list)
        M = desc_list[0].shape[1]

        ave_matrix = np.zeros((N, M), dtype=np.float64)

        for idx, structure_matrix in enumerate(desc_list):
            ave_matrix[idx] = structure_matrix.mean(axis = 0)

        return ave_matrix
