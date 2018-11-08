import dscribe
import numpy as np
import ase
from scipy.spatial.distance import cdist

class RematchKernel():
    """
    Rematch Kernel methods to go from local descriptors
    to global similarity
    """
    def __init__(self):
        return



    def compute_envkernel(self, local_a,local_b, gamma = 0.01):
        """Takes two matrices and computes the similarity
        based on the gaussian kernel.
        """
        return cdist(local_a,
                local_b,
                lambda u, v: np.exp(-gamma * np.sqrt(((u-v)**2).sum())),
                )

    def get_all_envkernels(self, desc_list):
        """Takes a list of M x N matrices, where M is the number of atoms
        in the system and N is the number of features of the descriptor.
        The matrices can be of different sizes.
        Returns a dictionary of environment kernels with the keys (i,j)
        corresponding to the position in the list.
        """
        ndatapoints = len(desc_list)
        envkernel_dict = {}
        for i in range(ndatapoints):
            for j in range(ndatapoints):
                envkernel = self.compute_envkernel(desc_list[i], desc_list[j])
                envkernel_dict[(i,j)] = envkernel
        return envkernel_dict

    def get_global_kernel(self, envkernel_dict, gamma, threshold):
        """Takes a dictionary of environment kernels with the keys (i,j)
        and m x n matrices as values (M: number of atoms in the system,
        N: number of features of the descriptor)
        Returns a squared matrix with the size of the given dataset.
        """
        keys = envkernel_dict.keys()
        envkernel_list = envkernel_dict.values()
        row_ids = np.array([key[0] for key in keys])
        col_ids = np.array([key[1] for key in keys])
        N, M = int(row_ids.max() + 1), int(col_ids.max() + 1)

        glosim = np.zeros((N, M), dtype=np.float64)
        for i,j in envkernel_dict:
            envkernel = envkernel_dict[i,j]
            glosim[i, j] = self.rematch(envkernel,
                gamma=gamma, threshold=threshold)
        return glosim



    def rematch(self, envkernel, gamma = 0.1, threshold = 1e-6):
        """
        Compute the global similarity between two structures A and B.
        It uses the Sinkhorn algorithm as reported in:
        Phys. Chem. Chem. Phys., 2016, 18, p. 13768
        Args:
            envkernel: NxM matrix of structure A with
                N and structure B with M atoms
            gamma: parameter to control between best match gamma = 0
                and average kernel gamma = inf.
        """

        n, m = envkernel.shape
        K = np.exp(-(1 - envkernel) / gamma)

        # initialisation
        u = np.ones((n,)) / n
        v = np.ones((m,)) / m

        en = np.ones((n,)) / float(n)
        em = np.ones((m,)) / float(m)

        Kp = (1 / en).reshape(-1, 1) * K

        # converge balancing vectors u and v
        itercount = 0
        error = 1
        while (error > threshold):
            uprev = u
            vprev = v
            v = np.divide(em, np.dot(K.T, u))
            u = np.divide(en, np.dot(K, v))

            # determine error every now and then
            if itercount % 5:
                error = np.sum((u - uprev) ** 2) / np.sum((u) ** 2) + np.sum((v - vprev) ** 2) / np.sum((v) ** 2)
            itercount += 1

        # using Tr(X.T Y) = Sum[ij](Xij * Yij)
        # P.T * C
        # P_ij = u_i * v_j * K_ij
        pity = np.multiply( np.multiply(K, u.reshape((-1,1))) , v)

        glosim = np.sum( np.multiply( pity, envkernel))

        return glosim
