DScribe
=======

.. image:: https://travis-ci.org/SINGROUP/dscribe.svg?branch=master
    :target: https://travis-ci.org/SINGROUP/dscribe

.. image:: https://coveralls.io/repos/github/SINGROUP/dscribe/badge.svg?branch=master
    :target: https://coveralls.io/github/SINGROUP/dscribe?branch=master

DScribe is a Python package for transforming atomic structures into fixed-size
numerical fingerprints. These fingerprints are often called "descriptors" and
they can be used in various tasks, including machine learning, visualization,
similarity analysis, etc. To get started you can check the :doc:`basic tutorial
<tutorials/basics>`.

.. note::
   We are incrementally adding support for calculating the derivatives of
   descriptors with respect to the atom positions. In version **0.5.x** you can
   find an initial implementation for getting these derivatives for the SOAP
   descriptor. Also check out the :doc:`new tutorial
   <tutorials/machine_learning/forces_and_energies>` on predicting energies and
   forces using these derivatives.

Capabilities at a Glance
========================

DScribe currently includes the following descriptors:

  - :doc:`Coulomb matrix <tutorials/descriptors/coulomb_matrix>`
  - :doc:`Sine matrix <tutorials/descriptors/sine_matrix>`
  - :doc:`Ewald sum matrix <tutorials/descriptors/ewald_sum_matrix>`
  - :doc:`Atom-centered Symmetry Functions (ACSF) <tutorials/descriptors/acsf>`
  - :doc:`Smooth Overlap of Atomic Positions (SOAP) <tutorials/descriptors/soap>`
  - :doc:`Many-body Tensor Representation (MBTR) <tutorials/descriptors/mbtr>`
  - :doc:`Local Many-body Tensor Representation (LMBTR) <tutorials/descriptors/lmbtr>`

Check the tutorials for more information.

Go Deeper
=========
You can find more details in our open-access article: `DScribe: Library of
descriptors for machine learning in materials science
<https://doi.org/10.1016/j.cpc.2019.106949>`_

Documentation for the source code :doc:`can be found here <doc/modules>`. The
full source code with examples and regression tests can be explored at `github
<https://github.com/SINGROUP/dscribe>`_.

.. toctree::
    :hidden:

    install
    tutorials/tutorials
    API <doc/modules>
    contributing
    publications
    citing
    changelog
    about
