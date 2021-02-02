Changelog
=========
 - 0.5.0:
    - Added initial implementation for getting the derivatives of the SOAP
      descriptor with respect to atom positions. It supports numerical
      derivatives for any SOAP configuration, and supports analytical
      derivatives when using the GTO radial basis. Currently only dense output
      is available.

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
