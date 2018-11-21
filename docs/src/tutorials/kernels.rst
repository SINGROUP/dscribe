Building kernels for kernel based learning methods
==================================================

Kernel-based methods such as kernel-ridge regression and support-vector
regression require the construction of a pairwise-similarity matrix. There are
however different ways of building such a similarity matrix. A commonly used
way is to measure similarity :math:`\kappa(\vec{x}_1, \vec{x}_2)` between two
samples :math:`\vec{x}_1` and :math:`\vec{x}_2` with a gaussian kernel:

.. math::
   \kappa(\vec{x}_1, \vec{x}_2) = e^{-\gamma\lvert \vec{x}_1 - \vec{x}_2 \rvert^2}

There are also other standard kernels, some of which are listed e.g. in the
`sklearn-documentation
<https://scikit-learn.org/stable/modules/metrics.html>`_.

In some cases it might however be benefitial to define a custom way of
measuring similarity.

Kernels for comparing structures based on local features
--------------------------------------------------------
Measuring the similarity of two structures based on local features can benefit
from a custom kernel. Here we introduce some ways of measuring similarity in
these cases. For further reading, consider the original article:

`"Comparing molecules and solids across structural and alchemical space, Sandip
De, Albert P. Bartók, Gábor Cásnyi, and Michele Ceriotti, Phys. Chem. Chem.
Phys., 18, 13754-13769, 2016" <https://doi.org/10.1039/C6CP00415F>`_

Average kernel
--------------
The simplest approach is to average the local contributions into one average
vector, which can be compared with any standard kernel. For example the
:class:`.SOAP`-descriptor can provide the average vector directly by specifying
average=True in the constructor. These vector may be fed into any generic
purpose kernel model with any kernel function.

Best-match kernel
-----------------
TODO

REMatch kernel
--------------
The ReMatch kernel, lets you choose between the best match of local
environments and the averaging strategy. The parameter *gamma* determines the
contribution of the two whereas *gamma = 0* means only the similarity of the
best matching local environments is taken into account and *gamma* going
towards infinite channels in the average solution.

.. code-block:: python

    from dscribe.utils import RematchKernel

    rematch = RematchKernel()
    envkernels = rematch.get_all_envkernels([soap_water, soap_methanol])
    remat = rematch.get_global_kernel(envkernels, gamma = 0.1, threshold = 1e-6)

    print(remat)


