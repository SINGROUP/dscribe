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
       :class:`.Descriptor`-class.
    5. Create new test module for the descriptor with the same module name, but
       in *dscribe/regtests/*. The tests should be implemented within a new
       class that inherits from the :class:`.TestBaseClass`-class found in
       *dscribe/regtests/*. Add the new tests as part of
       *dscribe/regtests/testrunner.py*.
    6. Ensure that your new tests and the existing tests run succesfully. You
       can run the tests by first installing the developer dependencies by
       running :code:`pip install -r dscribe/devrequirements.txt`, and then
       running *dscribe/regtests/testrunner.py*.
    7. Create tutorial for the descriptor in *dscribe/docs/src/tutorials*.
       Follow the structure of the existing tutorials.
    8. Create a pull request in GitHub.

Code style guideline
--------------------
  - The code is designed to run on Python 3 only from version 0.2.8 onwards.
  - The code style is guided by PEP 8, but ignoring the following:

     - E123: Closing bracket does not match indentation of opening bracket’s line
     - E124: Closing bracket does not match visual indentation
     - E126: Continuation line over-indented for hanging indent
     - E128: Continuation line under-indented for visual indent
     - E226: Missing whitespace around operator
     - E501: Line too long (82 > 79 characters)
     - E401: Module level import not at top of file

  - Spaces over tabs. The indent size is 4 spaces.
  - Classes and functions should be documented by following the `Google style guide
    <http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_
    that can be interpreted by the `sphinx Napoleon-extension
    <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_
