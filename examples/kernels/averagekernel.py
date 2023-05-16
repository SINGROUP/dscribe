"""Demostrates how global similarity kernels can be built from local atomic
environments.
"""
from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel

from ase.build import molecule

# We will compare two similar molecules
a = molecule("H2O")
b = molecule("H2O2")

# First we will have to create the features for atomic environments. Lets
# use SOAP.
desc = SOAP(species=[1, 6, 7, 8], r_cut=5.0, n_max=2, l_max=2, sigma=0.2, periodic=False, crossover=True, sparse=False)
a_features = desc.create(a)
b_features = desc.create(b)

# Calculates the similarity with an average kernel and a linear metric. The
# result will be a full similarity matrix.
re = AverageKernel(metric="linear")
re_kernel = re.create([a_features, b_features])

# Any metric supported by scikit-learn will work: e.g. a Gaussian:
re = AverageKernel(metric="rbf", gamma=1)
re_kernel = re.create([a_features, b_features])
