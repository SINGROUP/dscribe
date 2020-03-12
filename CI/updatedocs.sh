# Build docs, copy to correct docs folder, delete build
cd ../docs/src
sphinx-apidoc -o ./doc ../../dscribe
make html
cp -a build/html/. ../dev
rm -r build

# Push changes to docs
cd ../..
git add ./docs
git commit -m "CI documentation build [skip ci]"
git push
