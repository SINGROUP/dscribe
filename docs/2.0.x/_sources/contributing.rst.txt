Contributing
============

Follow these general instructions if you wish to contribute a new descriptor
implementation to DScribe:

    1. Contact the authors to discuss your idea for a new descriptor
       implementation.
    2. Read the code style guideline (below).
    3. Fork the repository and do all modifications within it.
    4. Implement the descriptor as a class within a new module in
       *dsribe/descriptors/*. The descriptor class should inherit from the
       :class:`.DescriptorGlobal` or :class:`.DescriptorLocal` class.
    5. Create new test module for the descriptor in *dscribe/tests/*.
       `pytest <https://docs.pytest.org/>`_ is used as the testing framework and
       several test utilities can be found in
       *dscribe/tests/conftest.py*. Ensure that your new tests and the existing
       tests run succesfully. You can run the tests by first installing the
       developer dependencies by running :code:`pip install -r
       dscribe/devrequirements.txt`, and then running :code:`pytest tests` from the
       repository root folder.
    6. Create tutorial for the descriptor in *dscribe/docs/src/tutorials*.
       Follow the structure of the existing tutorials.
    7. Create a pull request in GitHub.

Code style guideline
--------------------
  - We follow the `Black code style
    <https://black.readthedocs.io/en/stable/?badge=stable>`_,
    which is PEP 8 compliant. The good thing about Black is that you can
    simply run the autoformatter to ensure that you fullfill the code style.
    Before committing (or using pre-commit hooks), you should simply run the
    automatic formatting. Any unformatted code will be caught by the style
    checks in CI.
  - Classes and functions should be documented by following the `Google style guide
    <http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_
    that can be interpreted by the `sphinx Napoleon-extension
    <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_
