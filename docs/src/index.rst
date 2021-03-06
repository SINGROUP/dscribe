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
   Version **0.4.0** includes a fix to the layout of the SOAP feature vector.
   See issue `#48 <https://github.com/SINGROUP/dscribe/issues/48>`_ for more
   details. These changes break the backwards compatibility of the SOAP feature
   vectors. It is thus encouraged to start using version 0.4.0 or above if
   working with SOAP.

Capabilities at a Glance
========================

DScribe currently includes the following descriptors:

  - :doc:`Coulomb matrix <tutorials/coulomb_matrix>`
  - :doc:`Sine matrix <tutorials/sine_matrix>`
  - :doc:`Ewald sum matrix <tutorials/ewald_sum_matrix>`
  - :doc:`Atom-centered Symmetry Functions (ACSF) <tutorials/acsf>`
  - :doc:`Smooth Overlap of Atomic Positions (SOAP) <tutorials/soap>`
  - :doc:`Many-body Tensor Representation (MBTR) <tutorials/mbtr>`
  - :doc:`Local Many-body Tensor Representation (LMBTR) <tutorials/lmbtr>`

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
    about

Changelog
=========
 - 0.4.0:
    - Fixed an issue with the layout of the SOAP descriptor. The output size
      was incorrectly missing elements. See issue `#48 <https://github.com/SINGROUP/dscribe/issues/48>`_.
    - Added support for different averaging modes in SOAP. See issue `#44 <https://github.com/SINGROUP/dscribe/issues/44>`_.
    - Migrated completely from Cython to pybind11.

 - 0.3.5:
    - Added support for Python 3.8. See issue `#40 <https://github.com/SINGROUP/dscribe/issues/40>`_.

 - 0.3.2:
    - Improved performance for SOAP in combination with very large systems. See issue `#31 <https://github.com/SINGROUP/dscribe/issues/31>`_.

 - 0.2.8:
    - Removed support for Python 2.
