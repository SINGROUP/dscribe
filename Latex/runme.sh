#if the size of matrix is different, the size must be changed in the .lua file
lua ./convert.lua Raw/phi_raw.tex > Results/phi.tex
lua ./convert.lua Raw/phi_rawDX.tex > Results/phiDX.tex
lua ./convert.lua Raw/phi_rawDY.tex > Results/phiDY.tex
lua ./convert.lua Raw/phi_rawDZ.tex > Results/phiDZ.tex
