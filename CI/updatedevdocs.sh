version=development

# Build docs, copy to correct docs folder, delete build
cd ../docs/src
sphinx-apidoc -o ./doc ../../dscribe
make html SPHINXOPTS="-D version=$version -D release=$version"
cp -a build/html/. ../$version
rm -r build

# Push changes to docs
git config --global user.name "Azure Pipelines CI"
git config --global user.email "lauri.himanen@gmail.com"
cd ../..
git add ./docs
git commit -m "CI documentation build [skip ci]"
git push origin HEAD:$BUILD_SOURCEBRANCHNAME
