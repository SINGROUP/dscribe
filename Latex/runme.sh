#copy each matrix from the maxima results (including drivative,x ,y and z)
#which have to be generated each individually. currently the size is set to 21.
#if the size of matrix is different, the size must be changed in the .lua file
lua ./convert.lua Raw/phi_raw.tex > Results/phi.tex
lua ./convert.lua Raw/phi_rawDX.tex > Results/phiDX.tex
lua ./convert.lua Raw/phi_rawDY.tex > Results/phiDY.tex
lua ./convert.lua Raw/phi_rawDZ.tex > Results/phiDZ.tex
