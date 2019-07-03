# Update pip
pip install pip --update

# Install numpy before requirements, as we need a specific version for
# different python versions, and using a requirements file cannot ensure the
# installation of a specific version
if [$TRAVIS_PYTHON_VERSION <= "3.5"]
then
    pip install numpy==1.15.4
fi

# Development dependencies
pip install -r devrequirements.txt

# Compile CMBTR and ACSF extensions with cython. The .so files will be compiled
# during package setup
cythonize dscribe/libmbtr/cmbtrwrapper.pyx
cythonize dscribe/libacsf/acsfwrapper.pyx

pip install .
