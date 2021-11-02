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

#include <string>
#include <random>
#include <pybind11/numpy.h>
#include <Eigen/Dense>
#include "descriptorglobal.h"

namespace py = pybind11;
using namespace std;
using namespace Eigen;

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
            int seed
        );

        /**
         * For creating feature vectors.
         */
        void create_raw(
            py::detail::unchecked_mutable_reference<double, 1> &out_mu, 
            py::detail::unchecked_reference<double, 2> &positions_u, 
            py::detail::unchecked_reference<int, 1> &atomic_numbers_u,
            CellList &cell_list
        );

        /**
         * Get the number of features.
         */
        int get_number_of_features() const;

        /**
         * Calculate sorted eigenvalues.
         */
        void get_eigenspectrum(
            const Ref<const MatrixXd> &matrix,
            py::detail::unchecked_mutable_reference<double, 1> &out_mu
        );

        /**
         * Sort by row L2 norm, possibly introducing noise.
         */
        void sort(Ref<MatrixXd> matrix, bool noise);


        unsigned int n_atoms_max;
        string permutation;
        double sigma;
        int seed;

    private:
        mt19937 generator;
};
#endif
