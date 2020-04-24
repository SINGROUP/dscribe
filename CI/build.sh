# Update pip and setuptools
pip install pip --upgrade
pip install setuptools --upgrade

# Install cython separately
pip install cython

# Install development dependencies
cd ..
pip install -r devrequirements.txt

# Compile CMBTR and ACSF extensions with cython. The .so files will be compiled
# during package setup
cd dscribe/libmbtr/
cythonize mbtrwrapper.pyx -X language_level=3
cd ../libacsf/
cythonize acsfwrapper.pyx -X language_level=3

# Make a source distribution
cd ../..
python setup.py sdist

# Install the package itself from the newly created source distribution (this
# way the distribution validity is also checked)
version=`python setup.py --version`
cd dist
tar xvzf dscribe-$version.tar.gz
cd dscribe-$version
pip install .
