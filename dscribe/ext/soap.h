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
#include "descriptor.h"
#include "celllist.h"

namespace py = pybind11;
using namespace std;

/**
 * SOAP descriptor with GTO radial basis.
 */
class SOAPGTO: public Descriptor {
    public:
        /**
         *
         */
        SOAPGTO(
            double r_cut,
            int n_max,
            int l_max,
            double eta,
            py::dict weighting,
            bool crossover,
            string average,
            double cutoff_padding,
            py::array_t<double> alphas,
            py::array_t<double> betas,
            py::array_t<int> species,
            bool periodic
        );
        /**
         * For creating SOAP output.
         */
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> cell,
            py::array_t<bool> pbc,
            py::array_t<double> centers
        ) const;

        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> centers
        ) const;

        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> centers,
            CellList cell_list
        ) const;

        /**
         * Get the number of features.
         */
        int get_number_of_features() const;

        /**
         * Analytical derivatives.
         */
        void derivatives_analytical(
            py::array_t<double> derivatives,
            py::array_t<double> descriptor,
            py::array_t<double> xd,
            py::array_t<double> yd,
            py::array_t<double> zd,
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> cell,
            py::array_t<bool> pbc,
            py::array_t<double> centers,
            py::array_t<int> center_indices,
            py::array_t<int> indices,
            const bool attach,
            const bool return_descriptor
        ) const;

    private:
        const double r_cut;
        const int n_max;
        const int l_max;
        const double eta;
        const py::dict weighting;
        const bool crossover;
        const double cutoff_padding;
        const py::array_t<double> alphas;
        const py::array_t<double> betas;
        const py::array_t<int> species;
};

/**
 * SOAP descriptor with polynomial radial basis.
 */
class SOAPPolynomial: public Descriptor {
    public:
        /**
         *
         */
        SOAPPolynomial(
            double r_cut,
            int n_max,
            int l_max,
            double eta,
            py::dict weighting,
            bool crossover,
            string average,
            double cutoff_padding,
            py::array_t<double> rx,
            py::array_t<double> gss,
            py::array_t<int> species,
            bool periodic
        );
        /**
         * For creating SOAP output.
         */
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> cell,
            py::array_t<bool> pbc,
            py::array_t<double> centers
        ) const;

        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> centers
        ) const;

        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> centers,
            CellList cell_list
        ) const;

        /**
         * Get the number of features.
         */
        int get_number_of_features() const;

    private:
        const double r_cut;
        const int n_max;
        const int l_max;
        const double eta;
        const py::dict weighting;
        const bool crossover;
        const double cutoff_padding;
        const py::array_t<double> rx;
        const py::array_t<double> gss;
        const py::array_t<int> species;
};

#endif
