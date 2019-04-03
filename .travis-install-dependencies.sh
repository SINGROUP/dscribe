# Development dependencies
pip install -r devrequirements.txt

# Compile CMBTR and ACSF extensions with cython. The .so files will be compiled
# during package setup
cythonize dscribe/libmbtr/cmbtrwrapper.pyx
cythonize dscribe/libacsf/acsfwrapper.pyx

pip install .
