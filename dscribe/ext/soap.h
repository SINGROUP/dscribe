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

#ifndef SOAP_H
#define SOAP_H

#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;
using namespace std;

/**
 * SOAP descriptor.
 */
class SOAPGTO {
    public:
        /**
         *
         */
        SOAPGTO(
            double rCut,
            int nMax,
            int lMax,
            double eta,
            py::array_t<int> species,
            bool crossover,
            string average,
            double cutoffPadding,
            py::array_t<double> alphas,
            py::array_t<double> betas
        );
        /**
         * For creating SOAP output.
         */
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomicNumbers,
            py::array_t<double> centers
        ) const;

    private:
        const double rCut;
        const int nMax;
        const int lMax;
        const double eta;
        const py::array_t<int> species;
        const bool crossover;
        const string average;
        const float cutoffPadding;
        const py::array_t<double> alphas;
        const py::array_t<double> betas;
};

#endif
