from __future__ import absolute_import, division, print_function
from builtins import super
import math
import numpy as np
import itertools

from scipy.spatial.distance import squareform, pdist, cdist
from scipy.sparse import lil_matrix, coo_matrix
from scipy.special import erf

from describe.core import System
from describe.descriptors import Descriptor


class SOAP(Descriptor):
    """Implementation of the Many-body tensor representation up to K=3.

    You can use this descriptor for finite and periodic systems. When dealing
    with periodic systems, please always use a primitive cell. It does not
    matter which of the available primitive cell is used.
    """

    def __init__(
            self,
            atomic_numbers,
            k,
            periodic,
            flatten=True
            ):
        """
        """
        super().__init__(flatten)


    def describe(self, system):
        """Return the many-body tensor representation as a 1D array for the
        given system.

        Args:
            system (System): The system for which the descriptor is created.

        Returns:
            1D ndarray: The many-body tensor representation up to the k:th term
            as a flattened array.

        """
    myAlphas, myBetas = soaplite.genBasis.getBasisFunc(myrCut, Nmax) 
    soapMat = soaplite.get_soap_locals(atoms, Hpos, myAlphas, myBetas, rCut=myrCut, NradBas=Nmax, Lmax=myLMax, crossOver=True) 

    return soapMat


    def get_number_of_features(self):
        """Used to inquire the final number of features that this descriptor
        will have.

        Returns:
            int: Number of features for this descriptor.
        """




