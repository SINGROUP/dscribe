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

#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;
using namespace std;


void derivatives_soap_gto(
    py::array_t<double> dArr,
    py::array_t<double> positions,
    py::array_t<double> HposArr,
    py::array_t<double> alphasArr,
    py::array_t<double> betasArr,
    py::array_t<int> atomicNumbersArr,
    py::array_t<int> orderedSpeciesArr,
    py::array_t<int> displacedIndicesArr,
    double rCut,
    double cutoffPadding,
    int nAtoms,
    int Nt,
    int nMax,
    int lMax,
    int nCenters,
    double eta,
    bool crossover,
    string average);

#endif
