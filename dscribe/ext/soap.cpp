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
#include "soap.h"
#include "soapGTO.h"

using namespace std;

SOAPGTO::SOAPGTO(
    double rCut,
    int nMax,
    int lMax,
    double eta,
    bool crossover,
    string average,
    double cutoffPadding,
    py::array_t<double> alphas,
    py::array_t<double> betas
)
    : rCut(rCut)
    , nMax(nMax)
    , lMax(lMax)
    , eta(eta)
    , crossover(crossover)
    , average(average)
    , cutoffPadding(cutoffPadding)
    , alphas(alphas)
    , betas(betas)
{
}

void SOAPGTO::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<double> centers,
    py::array_t<int> atomicNumbers,
    py::array_t<int> orderedSpecies,
    int nAtoms,
    int nSpecies,
    int nCenters
) const
{
    soapGTO(
        out,
        positions,
        centers,
        this->alphas,
        this->betas,
        atomicNumbers,
        orderedSpecies,
        this->rCut,
        this->cutoffPadding,
        nAtoms,
        nSpecies,
        this->nMax,
        this->lMax,
        nCenters,
        this->eta,
        this->crossover,
        this->average
    );
}
