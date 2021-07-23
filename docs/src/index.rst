DScribe
=======

.. image:: https://dev.azure.com/laurihimanen/DScribe%20CI/_apis/build/status/SINGROUP.dscribe?branchName=master
    :target: https://dev.azure.com/laurihimanen/DScribe%20CI/_build/latest?definitionId=1&branchName=master

.. image:: https://coveralls.io/repos/github/SINGROUP/dscribe/badge.svg?branch=master
    :target: https://coveralls.io/github/SINGROUP/dscribe?branch=master

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

DScribe is a Python package for transforming atomic structures into fixed-size
numerical fingerprints. These fingerprints are often called "descriptors" and
they can be used in various tasks, including machine learning, visualization,
similarity analysis, etc. To get started you can check the :doc:`basic tutorial
<tutorials/basics>`.

.. note::
   Version **1.1.0** introduces new features and improvements to the SOAP
   descriptor: increased maximum lmax to 20 and added support for more
   customized weighting of the atomic density through weighting functions.
   Please check out the new :ref:`section in the SOAP tutorial
   <soap-weighting>` to learn more about weighting the atomic density. Also
   check out :doc:`the new tutorial on using DScribe for clustering
   <tutorials/machine_learning/clustering>`.

.. note::
   We are incrementally adding support for calculating the derivatives of
   descriptors with respect to the atom positions. From version **1.0.0**
   upwards you can find an implementation for getting derivatives of
   non-periodic systems for the SOAP descriptor. Please check the :doc:`new
   documentation on derivatives <tutorials/derivatives>` and the new :doc:`new
   tutorial <tutorials/machine_learning/forces_and_energies>` on predicting
   energies and forces using these derivatives.

Capabilities at a Glance
========================

DScribe currently includes the following descriptors:

.. list-table::
   :widths: 80 10 10
   :header-rows: 1

   * - Descriptor name
     - Features
     - Derivatives
   * - :doc:`Coulomb matrix <tutorials/descriptors/coulomb_matrix>`
     - ✓
     - 
   * - :doc:`Sine matrix <tutorials/descriptors/sine_matrix>`
     - ✓
     - 
   * - :doc:`Ewald sum matrix <tutorials/descriptors/ewald_sum_matrix>`
     - ✓
     - 
   * - :doc:`Atom-centered Symmetry Functions (ACSF) <tutorials/descriptors/acsf>`
     - ✓
     - 
   * - :doc:`Smooth Overlap of Atomic Positions (SOAP) <tutorials/descriptors/soap>`
     - ✓
     - ✓
   * - :doc:`Many-body Tensor Representation (MBTR) <tutorials/descriptors/mbtr>`
     - ✓
     - 
   * - :doc:`Local Many-body Tensor Representation (LMBTR) <tutorials/descriptors/lmbtr>`
     - ✓
     - 

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
    api
    contributing
    publications
    citing
    changelog
    about
