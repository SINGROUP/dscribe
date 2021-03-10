Changelog
=========
 - 1.0.0:
    - New features:
        - Added possibility to calculate the derivatives of the SOAP descriptor
          with respect to atom positions. For now, only non-periodic structures
          are supported. Supports numerical derivatives for any SOAP
          configuration, and analytical derivatives when using the GTO radial
          basis.
    - Breaking changes:
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
    - Fixes:
        - Fixed an issue with the layout of the SOAP descriptor. The output
          size was incorrectly missing elements. See issue `#48
          <https://github.com/SINGROUP/dscribe/issues/48>`_.
    - New features:
        - Added support for different averaging modes in SOAP. See issue `#44 <https://github.com/SINGROUP/dscribe/issues/44>`_.
    - Other:
        - Migrated completely from Cython to pybind11.

 - 0.3.5:
    - New features:
        - Added support for Python 3.8. See issue `#40 <https://github.com/SINGROUP/dscribe/issues/40>`_.

 - 0.3.2:
    - Improvements:
        - Improved performance for SOAP in combination with very large systems. See issue `#31 <https://github.com/SINGROUP/dscribe/issues/31>`_.

 - 0.2.8:
    - Other:
        - Removed support for Python 2.
