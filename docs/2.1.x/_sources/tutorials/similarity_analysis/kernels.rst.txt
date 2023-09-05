Building similarity kernels from local environments
===================================================

Measuring the similarity of structures becomes easy when the feature vectors
represent the whole structure, such as in the case of Coulomb matrix or MBTR.
In these cases the feature vectors are directly comparable with different
kernels, e.g. the linear or Gaussian kernel.

Local descriptors such as SOAP or ACSF can be used in the same way to compare
individual local atomic environments, but additional tools are needed to make
comparison of entire structures based on local environments.
This tutorial goes through two different strategies for building such global
similarity measures by comparing local atomic environments between structures.
For more insight, see :cite:`kernels`.

Average kernel
--------------
The simplest approach is to average over the local contributions to create a
global similarity measure. This average kernel :math:`K` is defined as:

.. math::
    K(A, B) = \frac{1}{N M}\sum_{ij} C_{ij}(A, B)

where :math:`N` is the number of atoms in structure :math:`A`, :math:`M` is the
number of atoms in structure :math:`B` and the similarity between local atomic
environments :math:`C_{ij}` can in general be calculated with any pairwise
metric (e.g.  linear, gaussian).

The class :class:`.AverageKernel` can be used to calculate
this similarity.  Here is an example of calculating an average kernel for two
relatively similar molecules by using SOAP and a linear and Gaussian similarity
metric:

.. literalinclude:: ../../../../examples/kernels/averagekernel.py
   :language: python
   :lines: 4-

REMatch kernel
--------------
The REMatch kernel lets you choose between the best match of local environments
and the averaging strategy. The parameter :math:`\alpha` determines the
contribution of the two: :math:`\alpha = 0` means only the similarity of
the best matching local environments is taken into account and :math:`\alpha
\rightarrow \infty` channels in the average solution. The similarity kernel
:math:`K` is defined as:

.. math::
    \DeclareMathOperator*{\argmax}{argmax}
    K(A, B) &= \mathrm{Tr} \mathbf{P}^\alpha \mathbf{C}(A, B)

    \mathbf{P}^\alpha &= \argmax_{\mathbf{P} \in \mathcal{U}(N, N)} \sum_{ij} P_{ij} (1-C_{ij} +\alpha \ln P_{ij})

where the similarity between local atomic environments :math:`C_{ij}` can once
again be calculated with any pairwise metric (e.g. linear, gaussian).

The class :class:`.REMatchKernel` can be used to calculate this similarity:

.. literalinclude:: ../../../../examples/kernels/rematchkernel.py
   :language: python
   :lines: 4-

.. bibliography:: ../../references.bib
   :style: unsrt
   :filter: docname in docnames
