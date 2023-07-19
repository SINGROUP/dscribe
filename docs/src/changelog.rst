Changelog
=========
 - 2.0.1:
    - Added:
        - Wheel distributions for Linux, MacOS and Windows.
    - Fixed:
        - Issue with exception not being raised when analytical derivatives were requested for SOAP with weighting turned on. See issue `#89 <https://github.com/SINGROUP/dscribe/issues/113>`_.

 - 2.0.0:
    - Added:
        - Added numerical derivatives for :code:`SineMatrix`, :code:`EwaldSumMatrix`, :code:`ACSF`, :code:`MBTR`, :code:`LMBTR` and :code:`ValleOganov`.
        - :code:`attach=True` is now supported when getting analytical :code:`SOAP` derivatives.
        - :code:`periodic=True` is now supported when getting numerical :code:`SOAP` derivatives.
        - Added analytical derivatives for :code:`MBTR` when using the :code:`smooth_cutoff` weighting and/or :code:`valle_oganov` normalization.
        - Added new tutorial on using descriptors for :doc:`visualization  <tutorials/visualization/coloring_by_environment>`
    - Fixed:
        - Fixed issue in all atoms not being taken into account when using the :code:`smooth_cutoff` weighting of :code:`MBTR`.
    - Changed:
        - Renamed the :code:`positions` argument of local descriptors to :code:`centers`.
        - The :code:`MBTR`, :code:`LMBTR` and :code:`ValleOganov` interfaces has been simplified so that only one term can be calculated at once.
    - Removed:
        - "unflattened" outputs from :code:`MBTR`, :code:`LMBTR`, :code:`CoulombMatrix`, :code:`EwaldSumMatrix` and :code:`SineMatrix`.
        - The :code:`rcut`, :code:`nmax`, and :code:`lmax` options for :code:`SOAP`, use :code:`r_cut`, :code:`n_max` and :code:`l_max` instead.
        - The :code:`rcut` and :code:`gcut` options for :code:`EwaldSumMatrix`, use :code:`r_cut` and :code:`g_max` instead.

 - 1.2.3:
    - Added:
        - Analytical derivatives for :code:`MBTR`, thanks to `jarnlaak <https://github.com/jarnlaak>`_.
          Note that not all normalization options and geometry functions are
          supported.
        - :doc:`Tutorial <tutorials/visualization/coloring_by_environment>` on
          how to create visualizations by employing a mapping from descriptor
          feature space into colors.

 - 1.2.2:
    - Fixed:
        - An issue with :code:`CoulombMatrix` ordering using :code:`sorted_l2`
          that was introduced in version 1.2.0. See issue `#89 <https://github.com/SINGROUP/dscribe/issues/89>`_.
    - Deprecated:
        - The :code:`rcut`, :code:`nmax`, and :code:`lmax` options for :code:`SOAP`, use :code:`r_cut`, :code:`n_max` and :code:`l_max` instead.
        - The :code:`rcut` and :code:`gcut` options for :code:`EwaldSumMatrix`, use :code:`r_cut` and :code:`g_max` instead.

 - 1.2.0:
    - Added:
        - :doc:`Valle-Oganov descriptor <tutorials/descriptors/valleoganov>`
        - Numerical derivatives for :code:`CoulombMatrix`
        - Tensorflow implementation of :doc:`the force-field training
          tutorial <tutorials/machine_learning/forces_and_energies>`, courtesy
          of `xScoschx <https://github.com/xScoschx>`_
    - Fixed:
        - Memory leak with the :code:`SOAP` descriptor. See issue `#69
          <https://github.com/SINGROUP/dscribe/issues/69>`_.
        - Increased the numerical precision used for ACSF. See issue `#70
          <https://github.com/SINGROUP/dscribe/issues/70>`_.

 - 1.1.0:
    - Added:
        - Support for l_max <= 20 for the :code:`SOAP` GTO basis.
        - Support for weighting the gaussians contributing to the atomic
          density. For more details see :doc:`the updated SOAP tutorial
          <tutorials/descriptors/soap>` (and issues `#20
          <https://github.com/SINGROUP/dscribe/issues/20>`_, `#58
          <https://github.com/SINGROUP/dscribe/issues/58>`_).
        - :code:`attach`-argument for the :code:`derivatives`-function (see issue `#63
          <https://github.com/SINGROUP/dscribe/issues/63>`_).

 - 1.0.0:
    - Added:
        - Possibility to calculate the derivatives of the :code:`SOAP` descriptor
          with respect to atom positions. For now, only non-periodic structures
          are supported. Supports numerical derivatives for any SOAP
          configuration, and analytical derivatives when using the GTO radial
          basis.
    - Changed:
        - The :code:`periodic` attribute now instructs the code to take into account
          the periodicity of the system as defined by the :code:`pbc`-attribute
          the :code:`ase.Atoms`. So in addition to setting
          :code:`periodic=True` in the descriptor, also make your system
          periodic in the wanted directions through the :code:`pbc` attribute.
        - The sparse output now uses the sparse matrices from the
          `sparse library <https://sparse.pydata.org/en/stable/>`_-library.
          This change is motivated by the need for n-dimensional sparse arrays
          in various places. See more at the :doc:`documentation page for
          sparse output. <tutorials/sparse>`
        - The output shapes have been made more consistent across different
          descriptors: global descriptors now produce 1D flattened output and
          local descriptors produce 2D flattened output for a single system.
          Whenever multiple systems are given, an additional dimension is added
          that runs across the different systems: for systems with the same
          number of atoms the output becomes a five-dimensional array,
          otherwise the output becomes a list of four-dimensional arrays.

 - 0.4.0:
    - Added:
        - Support for different averaging modes in :code:`SOAP`. See issue `#44 <https://github.com/SINGROUP/dscribe/issues/44>`_.
    - Fixed:
        - An issue with the layout of the :code:`SOAP` descriptor. The output
          size was incorrectly missing elements. See issue `#48
          <https://github.com/SINGROUP/dscribe/issues/48>`_.
    - Changed:
        - Migrated completely from Cython to pybind11.

 - 0.3.5:
    - Added:
        - Support for Python 3.8. See issue `#40 <https://github.com/SINGROUP/dscribe/issues/40>`_.

 - 0.3.2:
    - Changed:
        - Improved performance for SOAP in combination with very large systems. See issue `#31 <https://github.com/SINGROUP/dscribe/issues/31>`_.

 - 0.2.8:
    - Removed:
        - Support for Python 2.
