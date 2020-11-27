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
#include "soapGeneral.h"
#include <iostream>


void derivatives_soap_gto(
    py::array_t<double> dArr,
    py::array_t<double> cArr,
    py::array_t<double> positionsArr,
    py::array_t<double> centersArr,
    py::array_t<double> alphasArr,
    py::array_t<double> betasArr,
    py::array_t<int> atomicNumbersArr,
    py::array_t<int> orderedSpeciesArr,
    py::array_t<int> displacedIndicesArr,
    double rCut,
    double cutoffPadding,
    int nMax,
    int lMax,
    double eta,
    bool crossover,
    string average,
    bool returnDescriptor)
{
    int nSpecies = orderedSpeciesArr.shape(0);
    int nFeatures = crossover ? (nSpecies*nMax)*(nSpecies*nMax+1)/2*(lMax+1) : nSpecies*(lMax+1)*((nMax+1)*nMax)/2;
    auto d = dArr.mutable_unchecked<4>();
    auto displacedIndices = displacedIndicesArr.unchecked<1>();
    auto positions = positionsArr.mutable_unchecked<2>();
    auto centers = centersArr.unchecked<2>();
    int nCenters = centersArr.shape(0);

    // Calculate the desciptor value if requested
    if (returnDescriptor) {
        soapGTO(cArr, positionsArr, centersArr, alphasArr, betasArr, atomicNumbersArr, orderedSpeciesArr, rCut, cutoffPadding, nMax, lMax, eta, crossover, average);
    }
    
    // Central finite difference with error O(h^2)
    double h = 0.0001;
    vector<double> coefficients = {-1.0/2.0, 1.0/2.0};
    vector<double> displacement = {-1.0, 1.0};

    for (int iCenter=0; iCenter < nCenters; ++iCenter) {

        // Make a copy of the center position
        py::array_t<double> centerArr(3);
        auto center = centerArr.mutable_unchecked<1>();
        for (int i = 0; i < 3; ++i) {
            center(i) = centers(iCenter, i);
        }

        for (int iPos=0; iPos < displacedIndices.size(); ++iPos) {

            // Create a copy of the original atom position
            py::array_t<double> posArr(3);
            auto pos = posArr.mutable_unchecked<1>();
            for (int i = 0; i < 3; ++i) {
                pos(i) = positions(iPos, i);
            }

            for (int iComp=0; iComp < 3; ++iComp) {
                for (int iStencil=0; iStencil < 2; ++iStencil) {

                    // Introduce the displacement
                    positions(iPos, iComp) = pos(iComp) + h*displacement[iStencil];

                    // Initialize numpy array for storing the descriptor for this
                    // stencil point
                    py::array_t<double> cArr({1, nFeatures});

                    // Calculate descriptor value
                    soapGTO(cArr, positionsArr, centerArr, alphasArr, betasArr, atomicNumbersArr, orderedSpeciesArr, rCut, cutoffPadding, nMax, lMax, eta, crossover, average);
                    auto c = cArr.unchecked<2>();

                    // Add value to final derivative array
                    double coeff = coefficients[iStencil];
                    for (int iFeature=0; iFeature < nFeatures; ++iFeature) {
                        d(iCenter, iPos, iComp, iFeature) = (d(iCenter, iPos, iComp, iFeature) + coeff*c(0, iFeature));
                    }
                }
                for (int iFeature=0; iFeature < nFeatures; ++iFeature) {
                    d(iCenter, iPos, iComp, iFeature) = d(iCenter, iPos, iComp, iFeature) / h;
                }

                // Return position back to original value for next component
                positions(iPos, iComp) = pos(iComp);
            }
        }
    }
}

void derivatives_soap_polynomial(
    py::array_t<double> dArr,
    py::array_t<double> cArr,
    py::array_t<double> positionsArr,
    py::array_t<double> centersArr,
    py::array_t<int> atomicNumbersArr,
    py::array_t<int> orderedSpeciesArr,
    py::array_t<int> displacedIndicesArr,
    double rCut,
    double cutoffPadding,
    int nAtoms,
    int Nt,
    int nMax,
    int lMax,
    double eta,
    py::array_t<double> rwArr,
    py::array_t<double> gssArr,
    bool crossover,
    string average,
    bool returnDescriptor)
{
    int nFeatures = crossover ? (Nt*nMax)*(Nt*nMax+1)/2*(lMax+1) : Nt*(lMax+1)*((nMax+1)*nMax)/2;
    auto d = dArr.mutable_unchecked<4>();
    auto displacedIndices = displacedIndicesArr.unchecked<1>();
    auto positions = positionsArr.mutable_unchecked<2>();
    auto centers = centersArr.unchecked<2>();
    int nCenters = centersArr.shape(0);

    // Calculate the descriptor value if requested
    if (returnDescriptor) {
        soapGeneral(cArr, positionsArr, centersArr, atomicNumbersArr, orderedSpeciesArr, rCut, cutoffPadding, nAtoms, Nt, nMax, lMax, nCenters, eta, rwArr, gssArr, crossover, average);
    }
    
    // Central finite difference with error O(h^2)
    double h = 0.0001;
    vector<double> coefficients = {-1.0/2.0, 1.0/2.0};
    vector<double> displacement = {-1.0, 1.0};

    for (int iCenter=0; iCenter < nCenters; ++iCenter) {

        // Make a copy of the center position
        py::array_t<double> centerArr(3);
        auto center = centerArr.mutable_unchecked<1>();
        for (int i = 0; i < 3; ++i) {
            center(i) = centers(iCenter, i);
        }

        for (int iPos=0; iPos < displacedIndices.size(); ++iPos) {

            // Create a copy of the original atom position
            py::array_t<double> posArr(3);
            auto pos = posArr.mutable_unchecked<1>();
            for (int i = 0; i < 3; ++i) {
                pos(i) = positions(iPos, i);
            }

            for (int iComp=0; iComp < 3; ++iComp) {
                for (int iStencil=0; iStencil < 2; ++iStencil) {

                    // Introduce the displacement
                    positions(iPos, iComp) = pos(iComp) + h*displacement[iStencil];

                    // Initialize numpy array for storing the descriptor for this
                    // stencil point
                    py::array_t<double> cArr({1, nFeatures});

                    // Calculate descriptor value
                    soapGeneral(cArr, positionsArr, centerArr, atomicNumbersArr, orderedSpeciesArr, rCut, cutoffPadding, nAtoms, Nt, nMax, lMax, 1, eta, rwArr, gssArr, crossover, average);
                    auto c = cArr.unchecked<2>();

                    // Add value to final derivative array
                    double coeff = coefficients[iStencil];
                    for (int iFeature=0; iFeature < nFeatures; ++iFeature) {
                        d(iCenter, iPos, iComp, iFeature) = (d(iCenter, iPos, iComp, iFeature) + coeff*c(0, iFeature));
                    }
                }
                for (int iFeature=0; iFeature < nFeatures; ++iFeature) {
                    d(iCenter, iPos, iComp, iFeature) = d(iCenter, iPos, iComp, iFeature) / h;
                }

                // Return position back to original value for next component
                positions(iPos, iComp) = pos(iComp);
            }
        }
    }
}

//void derivatives(
    //py::array_t<double> dArr,
    //py::array_t<double> cArr,
    //py::array_t<double> positionsArr,
    //py::array_t<double> centersArr,
    //py::array_t<int> displacedIndicesArr,
    //CellList& cellList,
    //Descriptor descriptor,
    //bool returnDescriptor,
//)
//{
    //auto d = dArr.mutable_unchecked<4>();
    //auto positions = positionsArr.mutable_unchecked<2>();
    //auto displacedIndices = displacedIndicesArr.unchecked<1>();
    //auto centers = centersArr.unchecked<1>();
    //nFeatures = descriptor.get_number_of_features();

    //// Calculate the descriptor value if requested
    //if (returnDescriptor) {
        //descriptor.create(cArr);
    //}
    
    //// Central finite difference with error O(h^2)
    //double h = 0.0001;
    //vector<double> coefficients = {-1.0/2.0, 1.0/2.0};
    //vector<double> displacement = {-1.0, 1.0};

    //for (int iCenter=0; iCenter < nCenters; ++iCenter) {

        //// Make a copy of the center position
        //py::array_t<double> centerArr(3);
        //auto center = centerArr.mutable_unchecked<1>();
        //for (int i = 0; i < 3; ++i) {
            //center(i) = centers(iCenter*3+i);
        //}

        //for (int iPos=0; iPos < displacedIndices.size(); ++iPos) {

            //// Create a copy of the original atom position
            //py::array_t<double> posArr(3);
            //auto pos = posArr.mutable_unchecked<1>();
            //for (int i = 0; i < 3; ++i) {
                //pos(i) = positions(iPos, i);
            //}

            //for (int iComp=0; iComp < 3; ++iComp) {
                //for (int iStencil=0; iStencil < 2; ++iStencil) {

                    //// Introduce the displacement
                    //positions(iPos, iComp) = pos(iComp) + h*displacement[iStencil];

                    //// Initialize numpy array for storing the descriptor for this
                    //// stencil point
                    //py::array_t<double> cArr({1, nFeatures});

                    //// Calculate descriptor value
                    //descriptor.create(cArr, center);
                    //auto c = cArr.unchecked<2>();

                    //// Add value to final derivative array
                    //double coeff = coefficients[iStencil];
                    //for (int iFeature=0; iFeature < nFeatures; ++iFeature) {
                        //d(iCenter, iPos, iComp, iFeature) = (d(iCenter, iPos, iComp, iFeature) + coeff*c(0, iFeature));
                    //}
                //}
                //for (int iFeature=0; iFeature < nFeatures; ++iFeature) {
                    //d(iCenter, iPos, iComp, iFeature) = d(iCenter, iPos, iComp, iFeature) / h;
                //}

                //// Return position back to original value for next component
                //positions(iPos, iComp) = pos(iComp);
            //}
        //}
    //}
//}
