"""Demostrates how global similarity kernels can be built from local atomic
environments.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import (bytes, str, open, super, range, zip, round, input, int, pow, object)

from dscribe.descriptors import SOAP
from dscribe.kernels import AverageKernel

from ase.build import molecule

# We will compare two similar molecules
a = molecule("H2O")
b = molecule("H2O2")

# First we will have to create the features for atomic environments. Lets
# use SOAP.
desc = SOAP([1, 6, 7, 8], 5.0, 2, 2, sigma=0.2, periodic=False, crossover=True, sparse=False, normalize=True)
a_features = desc.create(a)
b_features = desc.create(b)

# Calculates the similarity with an average kernel and a linear metric. The
# result will be a full similarity matrix.
re = AverageKernel(metric="linear")
re_kernel = re.create([a_features, b_features])

# Any metric supported by scikit-learn will work: e.g. a Gaussian:
re = AverageKernel(metric="rbf", gamma=1)
re_kernel = re.create([a_features, b_features])
