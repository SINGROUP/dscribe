* 0579305 (HEAD -> tesseral, origin/tesseral) documented tesseral soap. next: document derivatives.
* 64bedb3 (origin/savingBadGTO, savingBadGTO) added stuff, changing soapGTO. Nest: change constants and functions to match imaginary spherical harmonics
* 3748bd7 checking l=3 for tesseral
| * 43dff60 (origin/predev) Fixed average flags.
| * 694538e Simplified tests.
| * 33f0b5d Fixed memory allocation issues.
| *   b51d9ad (origin/soapaverage) Merged.
| |\  
| | * 3a42bbc Fixes.
| | * cb4cbac Started to work on stratification for soapGeneral.
| | * e209023 Moved stratification to the C side of things for GTO.
| | * 3c86beb Simplified the final looping that creates the power spectrum in soapGeneral. Now both GTO and the general soap code look more similar.
| | * 9bc36fb Fixed SOAP symmetry issues in GTO basis.
| | * 09e9080 Initial gabug fix in GTO.
| | * c9f48ec Starting to fix gabug in GTO.
| | * b4a70eb Initial fix for the gabug.
| | * 71b17d5 Stashing.
| | * ab5d15d Getting somewhere.
| | * 8e5adf4 Started taking the stratification inside the C++ side.
| | * 75bbea3 Stashing.
| | | * a209ab3 (origin/development) deleted libmbtr
| | | * f75ed60 pybind mbtr
| | | * d58ac13 acsf pybind tests pass
| | |/  
| |/|   
| * | 02abf24 fixed soapGeneral inner average
| * |   077fe8a Merge branch 'soapaverage' of github.com:SINGROUP/dscribe into soapaverage
| |\ \  
| | * \   cab5845 Merge branch 'soapaverage' of github.com:SINGROUP/dscribe into soapaverage
| | |\ \  
| | * | | 0c6a9b6 Updated documentation.
| * | | | 830afef added soap GTO inner average
| | |/ /  
| |/| |   
| * | | ba50e63 debug soap inner
| * | | 299a321 Merge branch 'soapaverage' of github.com:SINGROUP/dscribe into soapaverage
| |\| | 
| | * | 52a4c1b Fixed typos.
| * | | 2f4e69d start soapGTO inner average
| |/ /  
| * |   83c10ac Merge branch 'temp' into soapaverage
| |\ \  
| | * | 38ff7dd Added test for SOAP inner averaging.
| * | | ce9e852 added inner average in soapGeneral
| * | | c0937e0 CI documentation build [skip ci]
| * | | 6e39aa0 Merge branch 'development' of github.com:SINGROUP/dscribe into development
| |\| | 
| | * | 9e6f059 CI documentation build [skip ci]
| * | | 63cf683 removed volume and angles from lattice
| * | | c4c548f added general regtests
| * | | f9b2335 removed cartesian product in utils geometry.py
| * | | b24ab8c added regtests for utils_species, descriptor base class and examples
| |/ /  
| * |   f24665c CI documentation build [skip ci]
| |\ \  
| |/ /  
|/| |   
* | | 6420fe2 (origin/master, origin/HEAD, master) CI documentation build [skip ci]
* | |   c97bf79 Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \ \  
| * | | 50333e2 CI documentation build [skip ci]
| * | | 91126bc Update azure-pipelines.yml for Azure Pipelines
| * | | 4e8147d CI documentation build [skip ci]
* | | | 9273a89 Added initial list of publications.
|/ / /  
* | |   aeb4cef Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \ \  
| * | | 1cdefde Update azure-pipelines.yml for Azure Pipelines
| * | | a884e70 CI documentation build [skip ci]
* | | | 41268df Fixed typo in readme.
|/ / /  
| * / 1a16586 CI documentation build [skip ci]
| |/  
| * f4497bc Started to test new spectrum.
| * c32b2c1 Fixed variable rename issue.
| * cb36a33 Cleaned up variable names and syntax.
| * bf5b9db Cleaned up variable names and syntax.
| * 5404419 Simplified soapGTO by merging two funtions together.
|/  
*   665e077 Merge branch 'azure-pipelines'
|\  
| *   2b70f10 Merge branch 'azure-pipelines' of github.com:SINGROUP/dscribe into azure-pipelines
| |\  
| | * 0348e44 (origin/azure-pipelines) CI documentation build [skip ci]
| * | 19ac978 Updated stable docs.
| |/  
| * 43f72b9 Fixed variable name.
| * 99a8b83 Trying out doc push.
| *   1320f90 Merge branch 'azure-pipelines' of github.com:SINGROUP/dscribe into azure-pipelines
| |\  
| | * fbcb58d Update azure-pipelines.yml for Azure Pipelines
| * | e96f5c8 Updated doc command.
| |/  
| *   f489d06 Merge branch 'azure-pipelines' of github.com:SINGROUP/dscribe into azure-pipelines
| |\  
| | * 6051b06 Update azure-pipelines.yml for Azure Pipelines
| * | 8f0786d Updated coverage command.
| * |   adadb21 Merge branch 'master' into azure-pipelines
| |\ \  
| | |/  
| |/|   
| * | 8e1dddf Updated coverage script.
| * | b93c554 Trying another way to include files.
| * | 0832acc Added source files to setup.py.
| * |   38ed05d Merge branch 'azure-pipelines' of github.com:SINGROUP/dscribe into azure-pipelines
| |\ \  
| | * | e3fb220 Update azure-pipelines.yml for Azure Pipelines
| * | | bcbe30c Updated build script to install package form pip source distribution.
| |/ /  
| * |   cef678b Merge branch 'azure-pipelines' of github.com:SINGROUP/dscribe into azure-pipelines
| |\ \  
| | * | 14e77a8 Update azure-pipelines.yml for Azure Pipelines
| * | |   29dc57b Merged with master.
| |\ \ \  
| | |/ /  
| |/| |   
| * | |   6484c97 Merge branch 'azure-pipelines' of github.com:SINGROUP/dscribe into azure-pipelines
| |\ \ \  
| | * | | 326bee9 Update azure-pipelines.yml for Azure Pipelines
| | * | | 4445d0f Update azure-pipelines.yml for Azure Pipelines
| * | | | cfd0aa2 Updated doc update script.
| |/ / /  
| * | |   2c34a5b Merge branch 'azure-pipelines' of github.com:SINGROUP/dscribe into azure-pipelines
| |\ \ \  
| | * | | 580d1ff Update azure-pipelines.yml for Azure Pipelines
| * | | | 0e39df5 Updated permissions.
| |/ / /  
| * | | e3b7172 Update azure-pipelines.yml for Azure Pipelines
| * | |   f355dad Merge branch 'azure-pipelines' of github.com:SINGROUP/dscribe into azure-pipelines
| |\ \ \  
| | * | | 71e8742 Update azure-pipelines.yml for Azure Pipelines
| | * | | 3027808 Update azure-pipelines.yml for Azure Pipelines
| * | | | ccc0039 Updated CI scripts.
| |/ / /  
| * | |   94bb5eb Merge branch 'azure-pipelines' of github.com:SINGROUP/dscribe into azure-pipelines
| |\ \ \  
| | * | | f923fdd Update azure-pipelines.yml for Azure Pipelines
| * | | | a7c36fe Modified path for build script.
| |/ / /  
| * | | 1df7e71 Changed pipeline config filepath.
| * | | de3dc07 Update azure-pipelines.yml for Azure Pipelines
| * | |   f6696ca Merge branch 'azure-pipelines' of github.com:SINGROUP/dscribe into azure-pipelines
| |\ \ \  
| | * | | f250e2c Update azure-pipelines.yml for Azure Pipelines
| | * | | d428668 Update azure-pipelines.yml for Azure Pipelines
| * | | | be385bb Moved things around.
| |/ / /  
| * | | e715956 Modified build stage.
| * | | 492d834 Update azure-pipelines.yml for Azure Pipelines
| * | | 522d9c7 Update azure-pipelines.yml for Azure Pipelines
| * | | 3652581 Update azure-pipelines.yml for Azure Pipelines
| * | | b67dc41 Update azure-pipelines.yml for Azure Pipelines
| * | | 08dca9a Update azure-pipelines.yml for Azure Pipelines
| * | | 71ace29 Update azure-pipelines.yml for Azure Pipelines
| * | | 8a2262c Set up CI with Azure Pipelines
* | | | 565e11e [skip travis] Travis documentation build: 418
| |_|/  
|/| |   
* | |   e857b5b Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \ \  
| * | | 9854b7e [skip travis] Travis documentation build: 417
* | | | 20f4239 Fixed typos in basic tutorial.
|/ / /  
* | | 69ace4e Added basic tutorial.
* | | 5290c81 [skip travis] Travis documentation build: 416
| |/  
|/|   
* | bfbc2f6 Added support for python 3.8.
* |   81f2356 Merge branch 'development'
|\ \  
| * | e492593 Added pybind11 as development dependency.
| * | 8f8002a Added python3.8 as test environment.
| * | bc49bd9 Re-cythonized wrapper files with newer version of Cython that supports python3.8.
* | | 1ab0faa [skip travis] Travis documentation build: 412
* | | f514736 Fixed doc build script.
* | | 27e3be2 Updated MANIFEST.in to include the header files for ACSF and (L)MBTR.
* | | 612d48e Fixed bug in unflattened LMBTR output, updated version numbers.
* | | f756f41 Merge branch 'development'
|\| | 
| * | f5c6378 Fixed bug in unflattened output of LMBTR k=2.
| |/  
* / 14052a3 Small fix in SOAP tutorial.
|/  
* 9edebcb Updated docs to include mention about issues with pybind11.
* d786845 Added vim folder to gitignore.
* ac2a5bd Added license header script.
* 702df86 Added performance tests.
* 8495ea6 Fixed issue with header files not ending up in the pip distribution. Had to specify them in MANIFEST.in, although this is not necessary for MBTR and ACSF...
* 6915a73 Updated documentation.
* b30a642 Updated docs, updated version.
*   cf5c0c3 Merge branch 'development'
|\  
| * baf637f Removed old code.
| * 04cbc06 Added regression tests for cell list when system is smaller than cutoff and when positions are outside.
| * ed54c09 Applied a fix to cell list implementation. some problems with systems that are smaller than the cutoff size.
* | b4c61de Updated dev docs.
|/  
* c5c9ef8 Fixed issue with local/global variable mixup in regtests.
* ed0e0c3 Weird issue with crossover, trying error tolerance.
* 5934a2d Corrected path in document build.
*   11219ff Merged with development, code corresponds to release 0.3.0.
|\  
| * 321a850 Updated documentation, added license notes, built documentation for 0.3.0.
| * 0f84ed0 Finally the cell list is now also implemented for polynomial rbf.
| * 2cf2cba Backup before enabling new cell list implementation for polynomial basis.
| * c28c012 Enabled crossover setting for polynomial basis.
| * 8ab3d5e The pybind11 for polynomial basis is now finally working.
| * d7fc135 Initial binding code for polynomial basis.
* | 4493b16 Merge branch 'development'
|\| 
| * 9dbc69c Modified get_location to support crossover=False.
| * 66a6302 Implemented get_location() for SOAP.
| * e7f020f Updated the displacement tensor calculation to conform to updates in ase 3.19.0. Also made this ase version the new requirement in setup.py
| * 120215e Enabled some tests.
| * e671119 Reverted to the ctypes binding for soapGeneral for now.
| * aae8f82 Backup
| * 9e95aa1 Starting to figure things out for soapGeneral.
| * 3ba6530 Backup.
| * 66ea83c Fixing poly.
| * 877da73 Backup.
| * af79d4b Started implementing crossover option for polynomial basis.
| * f56028a Added a Python/C++ binding for the CellList result.
| * 28d4424 Added the new ext folder.
| * 0348e14 Cleaning up the code a bit, renamed lib to ext as lib seems to be reserved.
| * b0a0a77 Now the polynomial soap also uses the same radial cutoff padding as GTO.
| * 0e8f12f Finished wrapping general SOAP implemetation with pybind11, tests pass.
| * 6622a0b The SOAP implementation has a superfluous 5 angstrom padding that was removed recently. After removing it the padding tests now pass with greater accuracy.
| * 88d37ab Wrapping the SOAP gto basis with pybind11 now done. At the same time a lot of duplicate code was removed. All gto tests now pass.
| * a2be2e8 Fixed issue in modifying numpy data in the C++ extension for SOAP.
| * 2db80ec Now using pybind11 to transfer data to soap GTO, issues with getting the data correctly read.
| * db90f15 Modifying soap to use the new cell list implementation.
| * 9da6072 Implemented proper regression tests for the cell list, now passing.
| * db9e9d7 Started building regression tests for the cell list.
| * 3608dd9 Finished initial generalized implementation of the cell list.
| * fb86f34 Now successfully accessing raw numpy data by reference with pybind11.
| * 6d88eab Starting to modularize the cell list implementation for testing and reuse.
| * a5f0aea Starting to integrate and test the binning code.
| *   df3a81b Merge pull request #33 from anjohan/master
| |\  
| | * 9daf604 remove unnecessary include
| | * c86a0c4 fix issue with rcut > system size, cleaning
| | * 6f2af74 implemented binning for GTO, scales linearly and gives same results
* | | e600c3e Updated docs and improved doc building scripts.
| |/  
|/|   
* | 9bdfcd5 [skip travis] Travis documentation build: 357
|/  
* fe450d2 Final version of the padding. Decreased some of the accuracy limits to gain a smaller padding and more speed.
*   b2d10f5 Merge branch 'master' into development
|\  
| * fe4ddec [skip travis] Travis documentation build: 352
* | 60f9d71 Fixed issue with SOAP accuracy on periodic systems, added new test for it.
* | a9dd32d Reverted back to fixed 5 angstrom padding for SOAP.
| | * af33407 (origin/soappadding) Playing with the SOAP padding.
| |/  
| *   aaefbad Merge branch 'master' of github.com:SINGROUP/dscribe
| |\  
| | * 584ab71 [skip travis] Travis documentation build: 351
| * | da0c60d Updated citation page.
| |/  
| *   428ec2f Merge branch 'development'
| |\  
| |/  
|/|   
* | 1f233cd Update README.md
* | 8bb8898 Update README.md
* | 776585c Sizing
* | 45c3269 Sizing
* | f997b84 Trying to get logo nicely sized
* | d76400c Trying to get logo working
* | c234695 Improved readme.
| *   4365b7b Merge branch 'master' of github.com:SINGROUP/dscribe
| |\  
| | * 701b9f1 [skip travis] Travis documentation build: 342
| * | 0e9daa4 Updated citation.
| |/  
| *   fb97ae7 Merge branch 'development'
| |\  
| |/  
|/|   
* | 4ddc2aa Added proper tests for ACSF periodicity.
* | dd0ae69 Added check for doing certain optimizations based on what positions are requested and if the system is periodic.
* | ae58a27 Issue in numpy writeable flag. Could not reproduce TravisCI problem on local machine... Trying if copying the data array helps.
* | 2fb6bff Initial version of periodic ACSF now working, comes along with a speed boost for large systems (got rid of some unnecessary pair calculations when all atoms are not used as centers).
* | 89a02ac Starting to make ACSF periodic.
| * 087600d [skip travis] Travis documentation build: 337
| *   0a68de3 Merge branch 'master' of github.com:SINGROUP/dscribe
| |\  
| | * 7f18ab0 [skip travis] Travis documentation build: 336
| * | 45d72d1 Fixed issue in table formatting, added mention abou the variable x in MBTR tutorial.
| |/  
| * fb2eef1 Added kernel documentation files.
| *   a98744c Merged with development.
| |\  
| |/  
|/|   
* | e0cd43c Updated version number for pip build.
* | 578a9b4 Preparing release for 0.2.9. Removed code for python2 support (future package), updated setup.py to be more informative.
| * 0096277 Updated development version number.
| * ad1dc35 Removed unnecessary extension files from MANIFEST.in
| *   85c23b6 Merge pull request #26 from jan-janssen/master
| |\  
| | * 5d62bd6 Fix MANIFEST.in
| |/  
| * b340e58 [skip travis] Travis documentation build: 328
| *   116bdf3 Merge branch 'development'
| |\  
| |/  
|/|   
* | f946e00 Re-enabled SOAP integration tests.
* | 15f3caa Added exception for invalid normalization factors.
* |   615c530 Merge branch 'master' into development
|\ \  
* | | ddfcdf2 Increased the SOAP integration test range.
| | * 3257cf6 [skip travis] Travis documentation build: 325
| |/  
| *   53ccf13 Merge branch 'master' of github.com:SINGROUP/dscribe
| |\  
| | * 09fd74b [skip travis] Travis documentation build: 323
| * |   3b1fef2 Merge branch 'development'
| |\ \  
| |/ /  
|/| /   
| |/    
* | 16ba637 Finished first version of SOAPLite integration.
* | a966116 Moved the function for determining an extended system into the geometry module from where it can be used by several descriptors. It is now connected to SOAP and MBTR. While doing this fixed an old issue with non-cubic cells possibly not getting all the correct neighours included. Also now SOAP takes the width of the gaussian directly into account when extending the system, instead of using a fixed padding of 5 angstroms.
* |   c1365d5 Merge branch 'master' into development
|\ \  
* | | 0f4b426 Updated the REMatch kernel code and documentation.
* | | d70794c Updated contribution guide to mention python 3 only.
* | | 0fb5396 Simplified the MBTr weight modification based on cell indices for k=3.
* | | d906ab6 Cleaned up the soaplite interface module.
* | | 48fb795 Simplified soaplite interface file.
* | | d8052d6 Fixed issue using different radial grid.
* | | eec207e First merge of SOAPLite into DScribe.
* | | fd08730 First integration attempt for SOAPLite.
| | *   982e3f4 Merge branch 'master' of github.com:SINGROUP/dscribe
| | |\  
| | | * ea917f0 [skip travis] Travis documentation build: 320
| | |/  
| |/|   
| | * de7f28c Updated the ACSF tutorial to include pseudocode for output and a mention of the fact that the central species is not encoded.
| |/  
| * 61d2a5c Fixed issues with missing documentation, updated REMatch example and documentation.
| * 4092851 [skip travis] Travis documentation build: 311
| * c0c1637 Updated documentation to contain a note about the performance update in 0.2.8.
| * 1151b42 [skip travis] Travis documentation build: 310
| * ccbc3b1 Updated doc configuration.
|/  
* 4015c79 Updated docs and examples.
* f0a8241 Modified MBTR to use the more general extended system creator.
* 0ae67f9 Updates on the building extended system.
* 6565a2c Removed some dead code.
* 227481e Final things before testing.
* 226660d Final things for creating the extended system, for now.
* 07d2831 Simplified extended system creation.
* 54ddd3d Some cleaning around.
* 21f671b Changed LMBTR k3 to use better extended system creator.
* 1b15240 Started fixing extended system creation.
* c92407b Added default setting for positions in LMBTR that is dientical to SOAP and ACSF.
* 278a69a Fixed issue with positional argument used in wrong place in ACSF, reintroduced a set of tests for LMBTR.
* 97511c7 Fixed issues in LMBTR unflattened output.
* 5ea0ba5 Quite finished LMBTR refactoring. Still needs testing.
* 272587c Beginning full fix for LMBTR.
* a3f3096 Starting to redo LMBTR.
* b101122 Starting to update LMBTR to accomodate to the optimized MBTR code.
* 74f6a64 Rewrote the tests to use the final output instead of the intermediate weights and geometry values.
* ca47b21 Removed a bunch of unused code from MBTR, enabled most tests.
* 827d5a4 Fixed issue with MBTR not using the correct weighting in the optimized version.
* 3ddd019 Trying to figure out the small differences in output.
* 8d44621 MBTR is now so fast that it is scary.
* 6b374c6 Removed chronic usage.
* abe8ac8 Removed chronic import.
* f17a9ef Enabled old regtests.
* 297e20c Cleaned up now obsolete parts of code.
* ac114de Introduced the same kd-tree inspired optimization to MBTR, addd a bunch of const declarations.
* c170ee4 Final optimization round for ACSF: removed some unnecessary ifs, fixed issue with distance matrix containing zeros for non-connected atoms.
* 0fab230 Switched to unordered map in ACSF species mapping.
* 970bca1 Refactored the ACSF code a bit.
* 547dad3 Switched to references in the index loops.
* 07fa20d Updated ACSF to use the dense adjacency matrix as accessing it seems to be much faster on the C++ side. It is stilled filled by using the cutoff.
* 03948e1 Second iteration of ACSF optimizations.
* a128769 First iteration of ACSF optimizations.
* ab3a96d Removed unnecessary test.
*   90dcec0 Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * 884794f [skip travis] Travis documentation build: 281
* | 977c1a9 Added a clipping for the cosine values in MBTR k=3 to ensure that the values stay in the range -1 < x < 1 even if there is some numerical noise.
|/  
*   da8275c Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * 182afd1 [skip travis] Travis documentation build: 280
* | edfff77 Added the sine matrix tutorial.
|/  
*   f3dac5b Fixed merge conflict.
|\  
| * e243e69 Updated installation documentation.
| * 12ebc49 Updated the Travis python 3.7 configuration.
| * 71f3956 Now Travis testing is done on 3.5, 3.6 and 3.7.
| * e76dcff Trying to get the numpy installation working correctly on Travis+Python3.4
| * 51716e7 Trying to get the numpy installation working correctly on Travis+Python3.4
| * 20ef7f8 Trying to get the numpy installation working correctly on Travis+Python3.4
| * ce0bd43 Trying to get the numpy installation working correctly on Travis+Python3.4
| * 07c351e Fixed issue with pip python_version.
| * 0be8d1c Added pip updation to travis setup.
| * 21f6382 Updated setup.py and travis setup for python>3.
| * e39ad4a Since numpy and many other dependencies are now dropping python 2 support, the codebase is no longer being tested on python 2.X, and the code will slowly migrate to python 3.X only. The release 0.2.7 will be the last official release that will be tested to work on python 2.X. The python 2.7 tests are now replaced with python 3.4 tests.
| * 228e050 Added the normalization with respect to number of atoms.
* | 307b086 [skip travis] Travis documentation build: 267
* | 4dba90f Updated the stable version to 0.2.7 with improved macOS support.
|/  
*   c527da4 Merge branch 'master' into development
|\  
| * 57f8f0f [skip travis] Travis documentation build: 262
| * 864d80f Added common issues to the installation page.
| * 52f1dc1 Added common issues to the installation page.
| *   1432734 Merge branch 'master' of github.com:SINGROUP/dscribe
| |\  
| | * 331d6a6 [skip travis] Travis documentation build: 260
| * | a23a423 Updated readme to include LMBTR.
| |/  
| *   067f284 Merged.
| |\  
| | * e5aaf50 [skip travis] Travis documentation build: 259
| * | f87b5c3 Updated documentation, switched to showing the stable documentation by default.
| |/  
| *   1b7dd19 Merged with development branch.
| |\  
| * \   3c00317 Merge branch 'master' of github.com:SINGROUP/dscribe
| |\ \  
| | * | 89a76f8 [skip travis] Travis documentation build: 250
| * | | 2c6a1a1 Updated setup to use older version of sklearn. The newest version does not support python 2.X anymore. Possibly we miht consider migrating to python3 only in DScribe as well.
| |/ /  
| * | c59a4da Updated version number for the quickfix.
| * |   f32b739 Merge branch 'master' of github.com:SINGROUP/dscribe
| |\ \  
| | * | ace6046 [skip travis] Travis documentation build: 248
| * | | cb4adca Quick fix to the MBTR performance. A too big cell size was used.
| |/ /  
* | | ebfb944 Trying out a more sensible fix for the C++11 problems with macOS. Solution inspired by MDAnalysis.
* | | 342c195 Introduced a possible fix for the problem with macOS compiling the C++11 extensions.
| |/  
|/|   
* | 3b5ae9b Added the stable version documentation.
* | 797b6e4 Added LMBTR tutorial, started adding documentation versioning.
* | fe3a74c Updated tests to conform to new defaults.
* | 6608e76 Updated the SOAP and MBTR documentation.
* | 6cd43d7 Updated the species tests.
* | 48d0d86 (origin/mbtrgeomfunc) Added new tests for the numpy positions in SOAP.create.
* | 0cdf4d2 Updated the SOAP documentation, removed the atomic_numbers parameter.
* | 61bb191 Modified tests.
* | f6cb079 Simplified the mbtr setup (decided to remove the separate classes for storing the setup, as there no longer is need to track the changes in the number ofgrid points, and because very heavy pre-testing of the parameters was removed.
* | 3d1290a Updated the MBTR regtests, added new classes to track the validity of the MBTR setup.
* | c3b9bba Modified the get_location function to return python-slices, enabled and improved the LMBTR regressions tests.
* | f55e771 Updated the LMBTR documentation and regression tests, simplified the way changes in the grid settings are tracked so that MBTR and LMBTR are more easily copiable and picklable.
* | 8d04312 Updated MBTR documentation, added analytical formula for calculating the number of features in LMBTR, removed the option for virtual positions in favour of the more simplistic is_center_repeated.
* | 84a8a27 Fixed issue in using numpy arrays as SOAP positions.
* | aa73097 First attempt at reducing LMBTR output size.
* | f06ab2b Started rehauling the MBTR tests.
* | 3a5d8ea Updated the example to show the new distance-based geometry function.
* | 5d700a1 Initial working merge and prototype for the new MBTR interface.
* | 78a9edf Merged with master.
|\| 
| *   60611d5 Merge branch 'master' of github.com:SINGROUP/dscribe
| |\  
| | * bce0a8c [skip travis] Travis documentation build: 246
| * |   518b8f8 Merge branch 'development'
| |\ \  
| | |/  
| |/|   
| | * dcde6f2 Updated the build flags for Mac, updated version number.
| * | 12cc115 Removed unused import in ACSF example.
| |/  
| * aba3e36 Fixed integer checking in elementaldistributions.
| * 8dd74b4 Added licence template to the library code, fixed issue with undefined MBTR normalization.
| *   aa2fa4f Merge branch 'master' into development
| |\  
| | * 80acb41 Fixed issue in calculating soap with crossover=False.
| * | 7c60333 Added new MBTR examples for normalization, periodic systems and finite systems.
| * | 7a91531 Fixed normalization initialization in LMBTR.
| * | 6ef1d88 Added new normalization options and associated tests for MBTR.
| * | b38fbbc Fixed issue MBTR related to non-equivalent spectras for periodic systems. This issue was caused ultimately by a wrong index used in extending the system and the wrongly handled weighting in the k3 term.
* | | 6ae3211 Started to refactor the MBTR interface, added new geometry function for distances and angles.
| |/  
|/|   
* | 8e22242 [skip travis] Travis documentation build: 238
* |   debf0bc Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \  
| * | 0d55407 [skip travis] Travis documentation build: 236
* | |   a88d86a Merge branch 'development'
|\ \ \  
| |/ /  
|/| /   
| |/    
| * bff71db Updated documentation on funding, updated the instructions for citing, updated css style.
* |   3f2160a Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \  
| * | f5eec50 [skip travis] Travis documentation build: 233
* | |   664abb7 Merge branch 'development'
|\ \ \  
| |/ /  
|/| /   
| |/    
| * 4684fe6 Updated documentation.
| * 0878274 Renamec Ewald matrix to Ewald sum matrix, added tutorial.
* |   a9ffd73 Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \  
| * | ec3cb03 [skip travis] Travis documentation build: 232
* | | 1810af8 Updated ACSF and SOAP documentation.
|/ /  
* | 510f8e8 Fixed SOAP create-method docstring.
* |   13a69a5 Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \  
| * | 158c8d8 [skip travis] Travis documentation build: 230
| |/  
* / 746b390 Updated MANIFEST.
|/  
* a5cc2a8 Updated version number in docs.
* 59a4058 Updated contribution guide.
* d68603a Fixed wrong file in SOAP tutorial.
* 585c528 Updated CM and SOAP documentation.
* 77f5895 Fixed issue in dynamically non-flattened, non-array output, added tests for creating multiple MBTR with non-flattened output.
* d8023d9 Fixed issues with creating non-flattened output for multiple systems.
* 3b568a7 Updated CM tutorial, removed references to create-batch.
* 013ac87 Now adding coding and copyright works.
* 80aae47 Improved documentation, starting to apply license header.
* c9a1c4d Defaulted to parallellization with processes, added more parallelization tests for each descripor.
* 06da8eb Fixed inconsistencies with matrix sizing.
* 80f9f6b Fixed issues with wrong offset in parallel calculation.
* fa7f775 Fixed multiprocessing to use the loky backend.
* 4cf7a21 Restricted the joblib backend selection, improved parallelization documentation, added result reordering for threading-based parallelization.
* 3c6e74c Updated SOAPLite dependency, added joblib dependency.
* 07a6669 Fixed the general tests.
* fef0607 (origin/batch) Reinstated SOAP tests.
* cabe10d Enabled parallelization for LMBTR.
* d93517a Enabled parallelization for SM.
* c444213 Enabled parallelization for EM.
* 4beea52 Enabled parallellization for CM.
* fd5ae7c Updated the MBTR C++ wrapper, added parallelization option to MBTR.
* a685174 Re-wrote ACSF in C++ and wrapped it with cython. Now it is picklable for parallization and the codebase is a bit easier to follow.
* e032aae Added possibility to control the joblib backend.
* 0246504 Improved documentation.
* b44b803 Updated the parallization of create() to be mostly in the superclass, subclasses need only to define the input, function and possibly the output size.
* 2f145df [skip travis] Travis documentation build: 212
* aedaec3 Updated version number.
* 8ed1977 [skip travis] Travis documentation build: 211
* 0c4f489 Added test for the deprecated atomi_numbers-parameter.
* bfdab4c Added python 2/3 compatibility header to species-module, fixed use of species-argument in kernel tests.
* 1462f08 Updated examples and documentation to use the species parameter by default.
* 3f7eabd Simplified the species module, added regtest for symbol-to-number mapping.
* a32418e Implemented species parameter and property in LMBTR.
* 7b8b197 Updated MBTR to use species and the pspecies property.
* c26c87c Updated SOAP to support the species parameters and property.
* 79265c2 Added support for species argument and property in ACSF, added tests.
* 9f54d5d Fixed wrong assignment of ACSF parameters in the regtests, tests passed before because the swapped parameters were identical.
* be1a512 Downgraded the pymatgen dependency to older version in order to support python 2. Pymatgen is only used for regtesting, so it is not a problem.
*   bdd893e Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * 0ae55ee [skip travis] Travis documentation build: 205
* | b295e81 Reinstated tests for Ewald matrix now that pymatgen is only a develoment dependency.
|/  
*   d3ce0b9 Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * 81a2255 [skip travis] Travis documentation build: 204
* | 46f2286 Added smaller version of the logo with better interpolation, tried fixing the blinking of the sphinx sidebar when main content is not yet loaded.
|/  
*   0ce91cd Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * 4cdebcb [skip travis] Travis documentation build: 203
* | a0252e4 Added logo assets, updated logo.
|/  
* 997b534 Removed uses of normalization flag.
*   91d616b Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * 3157fa1 [skip travis] Travis documentation build: 200
* |   2bc7ded Merge branch 'development'
|\ \  
| |/  
|/|   
| * e589c14 Removed the normalization option completely from SOAP: it should not be used in most cases.
| * df2689d Removed unnecessary normalization when calculating averaged soap.
* | f2c69b4 Merge branch 'development'
|\| 
| * 75723a3 Fixed ACSF not being interpreted as flattened, fixed issue in pickling ACSF during batch create.
* |   494f0ff Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \  
| * | ea0f445 [skip travis] Travis documentation build: 197
* | | f68a31c Removed old kernel tests.
|/ /  
* |   486a620 Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \  
| * | 0c9c1dc [skip travis] Travis documentation build: 196
* | | 878a80f Fixed mistake in the average kernel documentation.
|/ /  
* |   416933a Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \  
| * | 7e85297 [skip travis] Travis documentation build: 195
* | | 5259c22 Updated version number.
|/ /  
* |   0853308 Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \  
| * | 706c20d [skip travis] Travis documentation build: 194
* | | cb004d4 Added support for calculating similarity of two different sets of features, added optimizations based on symmetric matrices, added regtest for similarity of sparse features and similarity of two different sets of features.
|/ /  
* |   a6b9b56 Merge branch 'master' of github.com:SINGROUP/dscribe
|\ \  
| |/  
|/|   
| * a3b4ade [skip travis] Travis documentation build: 191
| * f3bcc09 FIX: refactored undefined variable in get_system
| * 9fbb57b [skip travis] Travis documentation build: 190
| *   e897467 Merge pull request #10 from yashasvi-ranawat/accept_multi_class
| |\  
| | * 1ade44e ENH: get_system accepts calsses with Atoms base calss
| * | 497f71b [skip travis] Travis documentation build: 189
* | | b1386a9 Removed matplotlib dependency, added sklearn dependency.
* | | cb6feec Completely refactored the kernel building, introduced proper normalization, added tests, improved kernel building documentation.
|/ /  
* |   833fcbe Merge branch 'development'
|\ \  
| |/  
|/|   
| * 0b416e4 Merged two ifs for reduced indentation.
| * c3126f6 Added bibliography management to documentation, improved SOAP documentation.
* | d067a1d [skip travis] Travis documentation build: 186
|/  
* 094c224 Allowed rcut to be smaller than 1 on polynomial basis.
*   7238c3f Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * 237a6c9 Travis documentation build: 182
* | a49f85c Added Travis CI command to skip testing the documentation build made by Travis itself.
|/  
* d3829fe Fixed issues in the batch_create function, updated SOAP documentation.
*   cf97250 Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * e16ce98 Travis documentation build: 178
* | 7b0d050 Updated soaplite dependency.
|/  
* 46452b4 Removed stub test for soap poly, updated soaplite dependency.
*   bed03f9 Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * dafe750 Travis documentation build: 175
* | 21c7502 Fixed periodicity boolean in poly soap poly test.
|/  
* a0ae6ea Updated SOAPLite dependency number.
* 1e8ba2c Updated verion numer.
*   5962d78 Merge branch 'poly'
|\  
| * 2fd67b7 Updated descriptor so that some properties are currently private.
| * b1bf0db SOAP tests no enabled and pass for polyas well.
| * 3e3075c Updated tests.
| * 7d82106 Changes.
| * 248ddfa Added analytical evaluation of the orthonormalization coefficients for polynomial basis.
| * 2cdcf92 Polynomial branch works, but prefactor is missing and weights for the rbfs are numerically calculated.
| *   b7abc8f Merge branch 'master' into poly
| |\  
| * | f34ed5d Saving.
| * |   909e478 Merged to master branch with prefactor.
| |\ \  
| * | | eefa85f Added new tests for polynomial basis.
* | | | 955bd4a Travis documentation build: 172
* | | | 7a35983 Updated ACSF tests to use the new default cutoff of 6Å proposed by Jörg.
* | | | 7fddbf0 Added initial ACSF example and tutorial. Updated the ACSF cosntructor a bit.
| |_|/  
|/| |   
* | | 4de5007 Travis documentation build: 169
* | | 90f46cc Temporarily disabling Ewald tests agains pymatgen, because pymatgen installation not working correctly.
* | | b392a9a Now the cython compilation is run before installation on TravisCI.
* | | 6a61ffb Increased version number for pypi release.
* | | 0adc18a Added new setup compilation flag for macos to fix issues in building MBTR C++11 extension.
| |/  
|/|   
* | 3281fde Travis documentation build: 164
* | cab32bb Updated the SOAP dosctring, increased the number of statistics gathered for testing the matrix permutation with random sorting.
* | 9170092 Updated soaplite dependency number, updated version number.
* |   fa11afc Merge branch 'prefactor'
|\ \  
| * | fd36e1b Added the normalization factor from the Wigner D matrix to the SOAP tests.
| * | 1e7a6e6 Modified the SOAP integration test to cover also the full partial power spectrum.
| * | ced37a4 Added the prefactor from the SOAP articles. Added a simple test for it, still needs to be tested with mutliple elememnts.
| |/  
* / b30c176 Travis documentation build: 161
|/  
*   d97bad7 Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * 41555b4 Travis documentation build: 159
* | 8162e7b Added test for SOAP rbf orthonormality, added requirement for new SOAP version with slightly changed cutoff definition.
|/  
*   6fba312 Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * 526934c Travis documentation build: 157
* | 9bf7a43 Replaced eta with sigma in the SOAP setup, because the standard deviation is more natural to work with.
|/  
* 6ca8dba Lowered the number of lmax in the integration tests to allow travis to do the integration within time limit. The numerical integration speed should be improved in the future.
*   20428ba Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * 176f87e Travis documentation build: 154
* | 5f10e78 Updated the SOAP integration test to use real spherical harmonics, and to test up to l=9 with non-unity sigma and atom dislocation in every cartesian direction. Updated the SOAPLite dependency version number. Added some sanity tests for the SOAP constructor.
|/  
*   5e88b50 Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * d5fe07c Travis documentation build: 152
* | 6dd786b Temporarily disabled the control of the gaussian width for SOAP. Introduced test against numerically integrated SOAP values.
|/  
* 1ba4695 Fixed push command to use master branch.
* f85668b Updated Travis to only send coverage and build documentation on tests passing on master branch.
* a769e5c Updated the documentation building script to push directly to master branch on successful build.
* 22d17bf Updated MBTR docstring.
* b577545 Trying out automatic building.
* 57b4601 Updated docs, fixed doc-building script.
* 7ae750d Trying to trigger build.
* b583a76 Added run permissions on doc script.
* 8bddb2d Added citation page source, fixed doc scripts path.
* b45f0fe Updated documenation, experimenting with automatic doc build.
* 062b3f0 Re-enabled tests on master, removed OSX tests because python is available in TravisCI.
* d321758 Changed OSX CI settings.
* 224158f Tried new os setup in CI.
* be5b362 Enabled CI only on development branch, tried adding OSX as build environment, removed 'm'-flag from ACSF compilation.
* 78dc697 Renamed gaussian width to sigma, started working on numerical check for SOAP coefficients.
* 5b7723e Updated to the new SOAPLite version with fixed module naming. Added utf encoding specification to SOAP and ACSF files.
* c14216a Fixed SOAPLite version.
* 36fb287 Updated the SOAPLite requirement, added small note about the SOAP basis set in the SOAP class docstring.
* 7e83103 Added the gaussian width parameter to SOAP.
* ecf80c0 Added support for gaussian width, implemented two different modes for SOAP, not giving same results at the moment.
*   97bc882 Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| * 02691ba small correction for documentation
* | 6cb1b75 Added build with sphinx 1.8.2, added initial MBTR documentation.
|/  
* f0bedc6 Started working on automatic doc building.
* 8e30a12 Added sphinx as a dvelopment dependency in hope of in future building doc automatically with Travis.
* 0556885 added coulombmatrix tutorial
* 926e2ef Added SOAP tutorial, added initial tutorial on kernels, added exception when creating SOAP for system that has undefined atomic numbers.
* cce2376 Fixed issue in LMBTR test not handling flattened output correctly.
*   783e02e Merge branch 'master' of github.com:SINGROUP/dscribe
|\  
| *   ca82b04 Merge branch 'master' of github.com:SINGROUP/dscribe
| |\  
| * | 65fad09 added example and tutorial for SOAP, fixed rematch_kernel example
* | | 3be9e29 Added correct sparse output for LMBTR.
| |/  
|/|   
* | 929c18e Fixed missing extension files in the pip packaging.
|/  
* aa7353b Fixed wrong package name in readme.
* 36902f0 Updated the documentation page look with logo and some css.
* 955cf75 Updated documentation.
* d0c0c02 Updated readme.
* 8b02808 Fixed coverage test issue.
* eb927be Increased number of repetitions in distribution test, increased version number.
* c18e790 Fixed issue in package name.
* d60f1d1 Updated documentation.
* 1f1c905 Renamed package to dscribe due to clash in PyPi.
* 1ffc343 Renamed package to dscribe due to clash in PyPi.
* d26a27f Refixed the wrapping problem in MBTR, the preivous fix was lost when merging code from LMBTR to MBTR.
* 6be5dac Fixed the integer type check in soap and lmbtr, impoved batch_create to allow the usage of positions for local descriptors.
* 3cf1212 Removed the requirement for grid information in MBTR contructor, because it may be input with the create_with_grid()-function.
* 7d68b75 Fixed flatten argument for SOAP (always true), fixed incorrect attribute naming on batch_create.
* 4eb3eb1 Improved the batch submission function, added new normalization and averaging options to SOAP.
* 593a0c3 Added the positions attribute to ACSF for calculating only specific positions.
* 13d6c74 Removed the n_atoms_max attribute from ACSF, because it is not needed due to the local nature of the descriptor. Fixed ElementalDistribution to conform to new structure.
* f74afb1 Added missing test file, removed the flatten attribute fromACSF, because the output is already flat, and flattening of different locations does not make sense.
* a9a97f1 Separated the matrix permutation tests to a different module, and added the missing test_sparse method.
* 3485722 Added new 'sparse' attribute to all descriptors that determines whether a sparse or a dense ouput is created, cleaned up ACSF tests, still need to activate many of the acsf tests after clearing some things.
* 5512338 Simplified the function layout, so that the for each descriptor the create()-function will not call nested functions and will have proper documentation. Refactored MBTR and LMBTR for virtual_positions and for reduced code overlap.
* fdfbd86 Fixed the ordering of the SOAP elements, added tests for the output vectors.
* 64ae830 Modified SOAP to support datasets with lots of different elements. The SOAP output is now returned as a sparse matrix that has been partitioned based on the interacting elements.
* 7fe50ba Added a new function for determining the correct atomic numbers to use in mbtr and lmbtr.
* ea27342 Improved the SOAP documentation by adding reference and fixing style errors. Improved SOAP tests by adding symmetry tests and other smaller ones.
* 9497cf9 Fixed the wrapping of atoms in the get_extended_system-function. Now there are no overlapping atoms.
* 1d29bb4 Updated MBTR documentation to specify the output types, removed chronic.
* 98b4148 Removed default grid settings, because they may lead to non-optimal results.
* ef32332 Fixed issue in the ordering of elements in the final MBTR vector. Added a new unit test to check for the same problem in the future.
* de24365 Updated the examples, added a new 'is_virtual'-parameter for local MBTR to determine whether the given position should correspond to a virtual or physical atom.
*   cce42b3 Modified the local MBTR implementation to work with the C++ code. Improved the lMBTR regression tests. The output of local MBTR was also made similar to regular MBTR in such a way that the outputs of these two are directly comparable.
|\  
| *   0d42516 Merge branch 'master' of github.com:SINGROUP/describe
| |\  
| | * 993062e added average kernel in utils and example
| * | a9092a0 Added the missing cutoff evaluation to K=3.
| |/  
* | 97a9dbd Added ensuring that only unique numbers in the given atomic numbers are used in MBTR.
* | 05cb830 Removed profiling script, improved error message on missing weighting parameters.
* | 271fca6 Added the double counting fix to MBTR, added proper tests for periodic systems for both k=2 and k=3.
* | e77f4ce Improved testing for MBTR.
* | 82f4f1b Fixed interface to K1, started rehauling regtests.
* | 1762fa5 Now also K1 is implemented in the C++ side consistently with other terms.
* | d85ae36 Unified the C++ interface for K2 and K3, simplified the python code based on new values returned from C++.
* | f4c5dd9 Fixing issue in getting correct unique angles for K3. Wrote better tests for angle calculation.
* | 0411cd0 Modified the interface for specifying the weighting function. The new interface allows for direct implementation if weighting in C++.
* | 17a59bc Improved code structure for C++ MBTR.
* | b92cac8 Created initial cython interface to the angle calculation.
* | c90845e Added better documentation for the C++ MBTR code, started implementing angle calculation.
* | 1f5e2a9 Merge branch 'master' into development
|\| 
| * 1ecabe3 Fixed constructor of LMBTR to use new normalization parameter.
| * f0550bc Added cython as a development requirement for travis.
| * 09a0086 Removed the unnecesary .so file for acsf extension: it will be compile upon setup.
| * d265a18 Removing mistakenly added test file.
| * 90b8a1b Added wrapper source file for enabling installation without cython, added cmbtr as an extension in setup.py, modified MBTR to include a constructor argument for enabling or disabling the normalization of Gaussians.
| * e3a3692 Temporarily adding cython as a development dependency to get the build running, will fix the setup so that cython is not needed.
| * 5dec7a2 Added chronic as a dependency, trying out if having cython in setup_requires does the trick.
| * c654ab5 Added Cython as a development dependency. Maybe later will try to figure out a way to not have to have cython installed for the user, because the cython dependency is not working correctly in setup.py (cython should be installed first before trying to install the package.
| *   eaa13fd Merging initial C++ implementation and python wrapper for MBTR. Currently only calcultes K2.
| |\  
| | * f4584ad Improved the Ewald matrix to contain more automatic determination of cutoffs.
| | * 3b39cef Added the extension building to setup.py, added cython as a dependency.
| | * 14291c8 Modified the tests to accommodate to the floating point precision used by the C++ code.
| | * b5665bb Implemented inverse distance calculation with C++. Seems to work, and twice as fast in creating the inverse distance map.
| | * bb3d115 Got the initial C++ wrapping for mbtr working, not much functionality present yet.
| | * badb859 Trying out a C++ implementation for the performance crititcal parts of MBTR.
| * | c07648d Update LICENSE
| * | 8b664ae Simplified the use of Ewald matrix significantly by providing sane defaults found from literature.
| |/  
| * 2f75ca4 Removed the unnecessary element_data subpackage, added symmetry checks to MBTR.
| * caf6d75 Fixed issue with basing an abstract class on the unittest.TestCase class.
| * cae6965 Added a testing base class that defines a test interface and has utitilies for common tests, such as invariance tests. Fixed an issue in random sorting (the cached norm vectors were used although system had changed), cleaned up the tests for Sine, Ewald, MBTR and Coulomb.
| * 53154a4 Fixed issue with LMBTR and the newly modified MBTR base class.
* | 49feae4 Started implementing MBTR k3 with C++.
| | *   31b370e (origin/grid) Merge branch 'master' into grid
| | |\  
| | |/  
| |/|   
| * |   37680f1 Merged.
| |\ \  
| | * | 1371d01 added coverage to LMBTR regtests
| | * | 6d41680 updated SOAP regtest
| | * | 9e89569 SOAP, LMBTR and MBTR ensures descriptor is updated before using it
| | * | 3539c65 added update method, minor dictionary improvements
| | * | 4a3f0d1 simplified SOAP get_number_of_features
| | * | a60fa64 simplified SOAP input of positions and atom indices in one list
| | * | 691de50 updated LMBTR regtests
| | * | 2490211 updated LMBTR example
| | * | 846bc76 simplified LMBTR input of positions and atom indices in one list
| * | | 0c15e09 Added the possibility of calculating MBTR with multiple grid values quickly by caching the scalars (distance, angles) but allowing the user to change the grid settings (sigma, min, max, n).
| |/ /  
| * |   401f794 Merge branch 'master' of github.com:SINGROUP/describe
| |\ \  
| | * | 6562a09 changed acsf argument types to atomic_numbers
| * | | ca10ac7 Fixed issue with periodic distances and non-orthorhombic cells. In some cases the reported distance was not correct. Also added a test for Coulomb matrix that ensures that non-periodic distances are used always.
| |/ /  
| * | 99c445b Improved MBTR docstrings.
| * | 1963861 fixed wrongly configured regtests for rematch_kernel
| * | c23b9b8 removed SOAP(all_atomtypes) to replace it with SOAP(atomic_numbers)
| * | e1b1f8a added rematch kernel, along with tests and example
| | * 94cd156 Added new modes: Coulomb and sphere for the GGrid descriptor.
| | *   a69649a Merged with master.
| | |\  
| | |/  
| |/|   
| * | 34a67b3 Added a new example for parallelly creating descriptors with a functional API similar to PySpark, updated also other parallel examples.
| * | e3bcd5c Fixed issue in LMBTR coming from new normalization parameter.
| * | 52c889d Added new normalization option to MBTR, improved the MBTR periodic tests.
| * |   055a065 Merge branch 'periodictests'
| |\ \  
| | * | d6b234e (origin/periodictests) added periodic tests for mbtr, commented out tests which failed in acsf and mbtr
| * | | 97312e0 extended LMBTR periodic regtest
| * | | cd1dd89 Merge branch 'periodictests'
| |\| | 
| | * | d064d55 added periodic tests to all descriptors but lmbtr
| * | | e10ae31 Made the distribution test to work also with any generic sigma value, not just with 1 that was previously hard-coded to the descriptor.
| * | | 6abb1b6 Fixed issue with the sigma value specified in the constructor not being used in the random sorting, tests now failing.
| * | |   e796a3b Merge branch 'master' of github.com:SINGROUP/describe
| |\ \ \  
| | * | | a3ab246 fix LMBTR rot n trans regtest
| | * | | 36008fb added lmbtr to testrunner
| | * | | 7c31e3a added python2.x headers
| | * | |   937a7a0 Merge branch 'test'
| | |\ \ \  
| | | |/ /  
| | |/| |   
| | | * | 20a778e added lmbtr regtests
| | | * | 99888dd fix check for atom_index
| | | * | 681f998 modified lmbtr doc-string
| | | * |   bb6a655 Merge branch 'test' of https://github.com/SINGROUP/describe into test
| | | |\ \  
| | | | * | 29d7716 LMBTR regtests: underconstruction
| * | | | | f7580cb Added test for checking that if sigma given, the random sorting should be also specified.
| |/ / / /  
| * | | | 36c30b1 Simplified the Ewald matrix calculation.
| * | | | f4d4887 Fixed Ewald matrix to contain only the pair terms, not the total energy of the subsystem.
| * | | |   33f756c Merge branch 'master' of github.com:SINGROUP/describe
| |\ \ \ \  
| | * \ \ \   1253605 Merge branch 'master' of github.com:SINGROUP/describe
| | |\ \ \ \  
| | * | | | | b254815 completed test_constructor, test_flatten, test_features and test_symmetries in acsf.py
| * | | | | | 5dd42f0 Testing Ewald matrix elements again.
| | |/ / / /  
| |/| | | |   
| * | | | | 749c01a Merge branch 'master' of github.com:SINGROUP/describe
| |\| | | | 
| | * | | |   6686be8 Merge branch 'acsftests'
| | |\ \ \ \  
| | | * | | | 80a02f6 (origin/acsftests) added g3  and g4 acsf value tests
| | * | | | | 4bc7e8b ACSF added G5 type
| | |/ / / /  
| | * | | | 21b5677 ACSF G3 bug fix
| | * | | | 083524c ACSF polish
| | * | | | f36765b ACSF removed useless C file
| | * | | | 2da67fe ACSF more polish
| | * | | |   28494c8 Merge branch 'master' of https://github.com/SINGROUP/describe
| | |\ \ \ \  
| | | * \ \ \   65a46ba Merge branch 'acsftests'
| | | |\ \ \ \  
| | | | * | | | d7e4c7f added number of features test for acsf
| | * | | | | | c836e6a ACSF bug fix + polish
| | |/ / / / /  
| | * / / / / 7c51869 ACSF bug fix
| | |/ / / /  
| * / / / / 1fb10bc Pushing initial documentation.
| |/ / / /  
| * | | | 61560c1 Trying a new stage setup for Travis.
| * | | | 211063e Playing arond with Travis stages to get coverage sent only from python 3.6.
| * | | | 71d8663 Added the future header also to test files.
| |/ / /  
| * | | 281a334 Removed osx from the os list as python is not yet supported on it.
| * | | a2a3364 Added OSX as a second OS to test on.
| * | | 02a6fcd Fooling around with the Travis config.
| * | | bff8316 Fixed issue with invalid parameter.
| * | |   f725e19 Merge branch 'master' of github.com:SINGROUP/describe
| |\ \ \  
| | * \ \   4a35f93 Merge branch 'master' of https://github.com/SINGROUP/describe
| | |\ \ \  
| | | * \ \   68a31fe Merge branch 'master' of github.com:SINGROUP/describe
| | | |\ \ \  
| | | * | | | a339a34 small addition
| | * | | | |   9bd33f2 Merge commit 'f4ff71f45e9'
| | |\ \ \ \ \  
| | | | |_|/ /  
| | | |/| | |   
| | | * | | |   f4ff71f resolved merge issue
| | | |\ \ \ \  
| | | | * | | | 4a60b3a added an update function in MBTR
| | | * | | | | fc95dc7 LMBTR takes multiple atom indices and/or positions
| | | * | | | | f5e33bf .create() passes kwargs, System accepts ghost atom
| | | * | | | | f037606 update function and cache checking to avoid calculations in MBTR
| | | |/ / / /  
| * | | | / / 925036f Enabled coverage for all python versions.
| | |_|_|/ /  
| |/| | | |   
| * | | | | 9c8bc6a Updated number of random samples.
| * | | | | 352184a Updated example.
| * | | | | 6dc45a3 Updated example.
| * | | | | 4c96270 Removed the file containing old random CM implementation.
| * | | | | 4673e44 Removed pathlib.
| |/ / / /  
| * | | | 6afb6f1 import SOAP in __init__.py
| * | | |   5463a53 Merge branch 'libacsf'
| |\ \ \ \  
| | * | | | 68d832b (origin/libacsf) included acsf c extensions in setyp.py and linked to correct location of .so file
| * | | | | e40de29 Made a test module for general tests, made a test module for ElementalDistribution.
| |/ / / /  
| * | | |   0e31489 Fixed merge conflict.
| |\ \ \ \  
| | | |/ /  
| | |/| |   
| | * | |   a09b126 merged soap
| | |\ \ \  
| | | * | | f17c106 (origin/akiSOAP) added soap.
| | | * | | bff5a12 added core
| | | |/ /  
| * | / / 53a4302 Added initial MBTR tests, fixed ACSF to load the .so file with correct path and fixed mixed tabs and spaces.
| |/ / /  
| * | | 37a41a3 Added sine matrix tests to the testsuite.
| * | |   093b14b Merge branch 'master' of github.com:SINGROUP/describe
| |\ \ \  
| | * \ \   9df599a Merge branch 'filippo'
| | |\ \ \  
| | | * | | 588eeb0 (origin/filippo) added ACSFs
| | | |/ /  
| | * | |   30ddc30 Merge branch 'master' of https://github.com/SINGROUP/describe
| | |\ \ \  
| | | * \ \   280746d Merge branch 'master' of github.com:SINGROUP/describe
| | | |\ \ \  
| | | * \ \ \   eb61998 Merge branch 'randsortcm'
| | | |\ \ \ \  
| | | | * | | | af46280 (origin/randsortcm) added tranlational and rotational symmetry test to CoulombMatrixTests
| | | | * | | | 3eaf19b added descriptions for RandomCoulombMatrix, as well as changed the descriptions adding sigma parameter.
| | * | | | | | 05ee705 removed non-asciis
| | | |_|_|/ /  
| | |/| | | |   
| * | | | | | f1d815b Added sine matrix tests.
| * | | | | |   9237159 Merged.
| |\ \ \ \ \ \  
| | | |_|_|/ /  
| | |/| | | |   
| | * | | | | 6d59507 Fixed issue in using flatten as sigma due to changes in MatrixDescriptor API.
| | * | | | | 6d6e081 Testing what is wrong with the tests.
| | * | | | | 7bd6be8 Made the code a bit more PEP8 compliant.
| | | |/ / /  
| | |/| | |   
| | * | | |   11b24dd Merge branch 'randsortcm'
| | |\ \ \ \  
| | | | |/ /  
| | | |/| |   
| | | * | | 3ff3652 added test to RandomCoulombMatrixTests, match with sorted cm
| | | * | | ef8924e added RandomCoulombMatrixTests class in regtests
| | | * | | ff14bd7 added sort_randomly method to matrixdescriptor class
| | | |/ /  
| * | / / b0be8ec Improved documentation.
| |/ / /  
| * / / b4d373f Added new tests for Ewald matrix: tests for the correct electrostatic energy.
| |/ /  
| * | 86c4d37 Fixing Python 2 compatibility issue.
| * | 5f8bd6a Started modifying Sine and Ewald matrices to conform to the new API, added initial simple tests.
| | * b3c615f Added the GGrid descriptor
| |/  
| * 9ad63f0 Simplified the descriptor hierarchy by mergin different ways of achieving permutational invariance as option in the constructor.
| * d33ed50 Added build bandges to readme.
| * 7653043 Testing Travis build.
| * 02e1820 Added an implementation of the Coulomb Matrix Eigevalue spectrum as a new descriptor, added more proper regression tests for the current CM implementations, added initial Travis CI script.
| | * 51575bb (origin/orbital) Added a functionin version of the orbital descriptor.
| |/  
| * e0d2e4a (origin/elemental) Fixed issues in not using the element occurrences in the ElementalDescriptor.
| * 07325b4 Corrected docstring.
| * 5948831 Modified the ElementalDistribution to handle both discrete and continuous distributions.
| *   d822a3e Merged with master.
| |\  
| | * 7367af8 Added initial implementation of elemental distribution, added better support for python 2.
| | * 8f69938 Updated readme with basic installation instructions.
| |/  
|/|   
* | b132d4c Removed some style issues in the local mbtr implementation.
* |   aed9f86 Merge branch 'master' into development
|\ \  
| * | 1414e6c Reverted the K0 implementation for now. Still exists in Vadim's branch.
* | | e73875b Added a proper example on how to create MBTR parallelly.
* | |   46494c2 Merge pull request #5 from yashasvi-ranawat/master
|\ \ \  
| * | | a1c1cbd Removed redundant codes in LMBTR: inherits from MBTR
| * | | ca0bbec Added descriptor: local MBTR
|/ / /  
| | * 3ddccbf Added a new class for elemental ditributions.
| | * ed8c453 MBTR descriptor extended with k0 term
| |/  
| * 8331a0d mbtr descriptor withadded k0 elemental descriptor term
|/  
*   f978b27 Merge branch 'master' of github.com:SINGROUP/describe
|\  
| * a15d4e8 Update LICENSE
* | bbaa93c Now the k-terms have to be explicitly give within a list or a set in order to avoid confusion. Fixed some old examples.
|/  
* a340dcc Added support for individual k terms, added regtests for this new feature.
*   868b7a5 Merge pull request #3 from yashasvi-ranawat/master
|\  
| * 49388b2 Added freedom to choose k terms based on octal notation Updated corresponding k value in example and README
|/  
* 6734cc5 Updated README example to reflect on newest changes.
* c483032 Fixed issue in getting the number of features for MBTR, added regression tests for getting number of features in MBTR, fixed MBTR example.
* 097217a Added normalization (output now normalized by dividing it with the maximum value of the gaussian that is used in the broadening of different K-terms), added a better way of calculating the discrete PDF of a Gaussian with the cumulative distribution function (the area under curve is more accurate with low number of sampling points), modified the grid input syntax to make it more explicit.
* 114c77e Fixed an indexing bug in the flattened version of MBTR, refactored System class to use ase.Atoms as base class, refactred the sorted descriptors to take advantage of inheritance.
* e0495a2 Added sorted sine matrix, made get_bumber_of_features to return int instead of float.
* 1e043f9 Added the sorted CoulombMatrix, added support for gathering statistics from ASE.Atoms and Systems.
* 1e252d9 Used the utils function in determining max number of atoms in parallel.py.
* d0284d8 Added an example on how to use multiple processors to create descriptors parallelly.
* 44788ed Started using coo_matrix instead of lil_matrix, noticing some speedup in MBTR creation.
* 2e34e54 Trying to fix readme markdown in github.
* 09c028d Trying to fix readme markdown in github.
* e3ece2b Improved the readme file.
* ef199d3 Fixed an interpretation problem in the K1 term (now the axis is the atomic number instead of count), fixed the centering problem of Gaussian smearing calcualated with convolution (now gaussian sum is calculated directly from the individual Gaussians), added support for ASE Atoms objects.
* 5efb029 Fixed a confusing example number in the docstrings.
* 128c974 Fixed bug in calculating the K2 distances.
* 829f6bd Fixed package names from 'crylearn' to 'describe'.
* baf2b6e Initial commit.
* 1528a09 Initial commit
