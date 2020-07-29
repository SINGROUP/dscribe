# Update pip and setuptools
pip install pip --upgrade
pip install setuptools --upgrade

# Install development dependencies
cd ..
pip install -r devrequirements.txt

# Make a source distribution
python setup.py sdist

# Install the package itself from the newly created source distribution (this
# way the distribution validity is also checked)
version=`python setup.py --version`
cd dist
tar xvzf dscribe-$version.tar.gz
cd dscribe-$version
pip install .
