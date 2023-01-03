#if the size of matrix is different, the size must be changed in the .lua file
lua ./convert.lua raw_phi.tex > phi.tex
lua ./convert.lua raw_phiDX.tex > phiDX.tex
lua ./convert.lua raw_phiDY.tex > phiDY.tex
lua ./convert.lua raw_phiDZ.tex > phiDZ.tex
