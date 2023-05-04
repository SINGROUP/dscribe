/*Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef SOAPGTO_H
#define SOAPGTO_H

#include <vector>
#include <pybind11/numpy.h>
#include "celllist.h"

namespace py = pybind11;
using namespace std;

inline int getCrosNum(int n);
inline int getDeltas(double* x, double* y, double* z, double *positions, double r[3], const vector<int> &indices);
inline void getRsZs(double* x,double* x2,double* x4,double* x6,double* x8,double* x10,double* x12,double* x14,double* x16,double* x18, double* y,double* y2,double* y4,double* y6,double* y8,double* y10,double* y12,double* y14,double* y16,double* y18, double* z,double* r2,double* r4,double* r6,double* r8,double* r10,double* r12,double* r14,double* r16,double* r18,double* z2,double* z4,double* z6,double* z8,double* z10,double* z12,double* z14,double* z16,double* z18, int size, int lMax);
void getAlphaBeta(double* aOa, double* bOa, double* alphas, double* betas, int Ns,int lMax, double oOeta, double oOeta3O2);
void getCfactors(double* preCoef, int Asize, double* x,double* x2, double* x4, double* x6, double* x8, double* x10,double* x12,double* x14,double* x16,double* x18, double* y,double* y2, double* y4, double* y6, double* y8, double* y10,double* y12,double* y14,double* y16,double* y18, double* z, double* z2, double* z4, double* z6, double* z8, double* z10,double* z12,double* z14,double* z16,double* z18, double* r2, double* r4, double* r6, double* r8,double* r10,double* r12,double* r14,double* r16,double* r18, int totalAN, int lMax);
void getC(double* CDevX,double* CDevY, double* CDevZ, double* C, double* preCoef, double* x, double* y, double* z,double* r2, double* bOa, double* aOa, double* exes,  int totalAN, int Asize, int Ns, int Ntypes, int lMax, int posI, int typeJ,vector<int>&indices);
void getP(double* soapMat, double* Cnnd, int Ns, int Ts, int Hs, int lMax);
void soapGTO(
    py::array_t<double> derivatives,
    py::array_t<double> descriptor,
    py::array_t<double> cdevX,
    py::array_t<double> cdevY,
    py::array_t<double> cdevZ,
    py::array_t<double> positions,
    py::array_t<double> centers,
    py::array_t<int> center_indices,
    py::array_t<double> alphasArr,
    py::array_t<double> betasArr,
    py::array_t<int> atomicNumbersArr,
    py::array_t<int> orderedSpeciesArr,
    const double rCut,
    const double cutoffPadding,
    const int Ns,
    const int lMax,
    const double eta,
    py::dict weighting,
    const bool crossover,
    string average,
    py::array_t<int> indices,
    const bool attach,
    const bool return_descriptor,
    const bool return_derivatives,
    CellList cell_list
);

#endif

