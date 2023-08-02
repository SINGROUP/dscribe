DScribe
=======

.. image:: https://github.com/SINGROUP/dscribe/actions/workflows/actions.yml/badge.svg
    :target: https://github.com/SINGROUP/dscribe/actions/workflows/actions.yml/badge.svg

.. image:: https://coveralls.io/repos/github/SINGROUP/dscribe/badge.svg?branch=master
    :target: https://coveralls.io/github/SINGROUP/dscribe?branch=master

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

DScribe is a Python package for transforming atomic structures into fixed-size
numerical fingerprints. These fingerprints are often called "descriptors" and
they can be used in various tasks, including machine learning, visualization,
similarity analysis, etc. To get started you can check the :doc:`basic tutorial
<tutorials/basics>`.

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
     - ✓
   * - :doc:`Sine matrix <tutorials/descriptors/sine_matrix>`
     - ✓
     - ✓
   * - :doc:`Ewald sum matrix <tutorials/descriptors/ewald_sum_matrix>`
     - ✓
     - ✓
   * - :doc:`Atom-centered Symmetry Functions (ACSF) <tutorials/descriptors/acsf>`
     - ✓
     - ✓
   * - :doc:`Smooth Overlap of Atomic Positions (SOAP) <tutorials/descriptors/soap>`
     - ✓
     - ✓
   * - :doc:`Many-body Tensor Representation (MBTR) <tutorials/descriptors/mbtr>`
     - ✓
     - ✓
   * - :doc:`Local Many-body Tensor Representation (LMBTR) <tutorials/descriptors/lmbtr>`
     - ✓
     - ✓
   * - :doc:`Valle-Oganov descriptor<tutorials/descriptors/valleoganov>`
     - ✓
     - ✓

Check the tutorials for more information.

Go Deeper
=========
You can find more details in the following articles:

  * `DScribe: Library of descriptors for machine learning in materials science <https://doi.org/10.1016/j.cpc.2019.106949>`_
  * `Updates to the DScribe Library: New Descriptors and Derivatives <https://doi.org/10.1063/5.0151031>`_

Documentation for the source code :doc:`can be found here <doc/modules>`. The
full source code with examples and tests can be explored at `github
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