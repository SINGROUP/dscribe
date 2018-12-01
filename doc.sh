# Used to build the sphinx documentation and push it to github directly after
# succesfull build.
cd docs/src
sphinx-apidoc -o ./doc ../../dscribe
make html
cp build/html/ ../
rm -r build
