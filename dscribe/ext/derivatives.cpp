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

#include "derivatives.h"
#include "soapGTO.h"
#include <iostream>


void derivatives_soap_gto(
    py::array_t<double> dArr,
    py::array_t<double> positionsArr,
    py::array_t<double> centersArr,
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
    string average)
{
    int nFeatures = crossover ? (Nt*nMax)*(Nt*nMax+1)/2*(lMax+1) : Nt*(lMax+1)*((nMax+1)*nMax)/2;
    auto d = dArr.mutable_unchecked<4>();
    auto displacedIndices = displacedIndicesArr.unchecked<1>();
    auto positions = positionsArr.mutable_unchecked<2>();
    auto centers = centersArr.unchecked<1>();
    
    double h = 0.0001;
    vector<double> coefficients = {-1.0/2.0, 0.0, 1.0/2.0};
    vector<double> displacement = {-1.0, 0.0, 1.0};

    for (int iCenter=0; iCenter < nCenters; ++iCenter) {

        // Make a copy of the center position
        py::array_t<double> centerArr(3);
        auto center = centerArr.mutable_unchecked<1>();
        for (int i = 0; i < 3; ++i) {
            center(i) = centers(iCenter*3+i);
        }

        for (int iPos=0; iPos < displacedIndices.size(); ++iPos) {

            // Create a copy of the original atom position
            py::array_t<double> posArr(3);
            auto pos = posArr.mutable_unchecked<1>();
            for (int i = 0; i < 3; ++i) {
                pos(i) = positions(iPos, i);
            }

            for (int iComp=0; iComp < 3; ++iComp) {
                for (int iStencil=0; iStencil < 3; ++iStencil) {

                    // Skip zero coefficient stuff
                    double coeff = coefficients[iStencil];
                    if (coeff == 0) {
                        continue;
                    }

                    // Introduce the displacement
                    positions(iPos, iComp) = pos(iComp) + h*displacement[iStencil];

                    // Initialize numpy array for storing the descriptor for this
                    // stencil point
                    py::array_t<double> cArr({1, nFeatures});

                    // Calculate descriptor value
                    soapGTO(cArr, positionsArr, centerArr, alphasArr, betasArr, atomicNumbersArr, orderedSpeciesArr, rCut, cutoffPadding, nAtoms, Nt, nMax, lMax, nCenters, eta, crossover, average);
                    auto c = cArr.unchecked<2>();

                    // Add value to final derivative array
                    for (int iFeature=0; iFeature < nFeatures; ++iFeature) {
                        d(iCenter, iPos, iFeature, iComp) = (d(iCenter, iPos, iFeature, iComp) + coeff*c(0, iFeature));
                    }
                }
                for (int iFeature=0; iFeature < nFeatures; ++iFeature) {
                    d(iCenter, iPos, iFeature, iComp) = d(iCenter, iPos, iFeature, iComp) / h;
                }
            }
            // After running through the components and stencils for this atom,
            // reset the position to it's original value
            for (int i = 0; i < 3; ++i) {
                positions(iPos, i) = pos(i);
            }
        }
    }
}
