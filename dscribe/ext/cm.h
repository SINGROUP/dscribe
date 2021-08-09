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

#ifndef CM_H
#define CM_H

#include <pybind11/numpy.h>
#include <string>
#include "descriptorglobal.h"

namespace py = pybind11;
using namespace std;

/**
 * Coulomb matrix descriptor.
 */
class CoulombMatrix: public DescriptorGlobal {
    public:
        /**
         * Constructor, see the python docs for more details about variables.
         */
        CoulombMatrix(
            unsigned int n_atoms_max,
            string permutation,
            double sigma,
            int seed,
            bool flatten
        );

        /**
         * For creating feature vectors.
         */
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> cell,
            py::array_t<bool> pbc
        ) const;

        /**
         * Get the number of features.
         */
        int get_number_of_features() const;

        void getEigenspectrum(
            py::array_t<double> out,
            py::array_t<double> matrix
        ) const;

        void sort(
            py::array_t<double> out,
            py::array_t<double> matrix
        ) const;

        void sortRandomly(
            py::array_t<double> out,
            py::array_t<double> matrix
        ) const;

    private:
        unsigned int n_atoms_max;
        string permutation;
        double sigma;
        int seed;
        bool flatten;
};
#endif
