Installation
============
The newest versions of the package are compatible with Python >= 3.6 (tested on
3.6, 3.7 and 3.8). DScribe versions <= 0.2.7 also support Python 2.7. We
currently only support Unix-based systems, including Linux and macOS. For
Windows-machines we suggest using the `Windows Subsystem for Linux (WSL)
<https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux>`_. The exact list
of dependencies are given in setup.py and all of them will be automatically
installed during setup.

The package contains C++ extensions that are automatically compiled during
install. On Linux-based systems the compilation tools are typically installed
by default, on MacOS you may need to install additional command line tools if
facing issues during setup (see common issues below).

The latest stable release is available through pip: (add the *-\\-user* flag if
root access is not available)

.. code-block:: sh

    pip install dscribe

To install the latest development version, clone the source code from
github and install with pip from local file:

.. code-block:: sh

    git clone https://github.com/SINGROUP/dscribe.git
    cd dscribe
    pip install .

Common issues
-------------
 - **fatal error: pybind11/pybind11.h: No such file or directory**: The package
   depends on pybind11 which is used for communication between python and the
   C++ extensions. Although pybind11 is specified as a requirement in setup.py,
   you may experience issues in pip correctly finding it. In this case you will
   need to install pybind11 before attempting to install dscribe by using the
   following command:

   .. code-block:: sh

       pip install pybind11

 - **fatal error: Python.h: No such file or directory**: The package depends on
   C/C++ extensions that are compiled during the setup. For the compilation to
   work you will need to install the *pythonX.X-dev*-package, where X.X is the
   python version you use to run dscribe. E.g. for python 3.7 on Ubuntu this
   package could be installed with:

   .. code-block:: sh

       sudo apt install python3.7-dev

 - **Installation errors on MacOS**: The package depends on C++ extensions that
   are compiled during the setup. If experiencing problems with setup on MacOS,
   you may need to install the Xcode Command Line tools package. This can be
   done with:

   .. code-block:: sh

       xcode-select —install
