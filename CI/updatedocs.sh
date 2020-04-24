# Build docs, copy to correct docs folder, delete build
cd ../docs/src
sphinx-apidoc -o ./doc ../../dscribe
make html
cp -a build/html/. ../dev
rm -r build

# Push changes to docs
git config --global user.email "lauri.himanen@gmail.com"
git config --global user.name "Azure Pipelines CI"
git checkout master
cd ../..
git add ./docs
git commit -m "CI documentation build [skip ci]"
git push origin HEAD:$(Build.SourceBranchName)
