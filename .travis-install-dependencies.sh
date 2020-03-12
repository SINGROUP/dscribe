# Update pip
pip install pip --upgrade

# Development dependencies
pip install -r devrequirements.txt

# Compile CMBTR and ACSF extensions with cython. The .so files will be compiled
# during package setup
cd dscribe/libmbtr/
cythonize mbtrwrapper.pyx -X language_level=3
cd ../libacsf/
cythonize acsfwrapper.pyx -X language_level=3

cd ../../
pip install .
