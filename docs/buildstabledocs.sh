# Build docs, copy to correct docs folder, delete build
git checkout master
cd docs/src
sphinx-apidoc -o ./doc ../../dscribe
make html
cp -a build/html/. ../
rm -r build
