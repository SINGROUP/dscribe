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

#ifndef DESCRIPTORGLOBAL_H
#define DESCRIPTORGLOBAL_H

#include <pybind11/numpy.h>
#include <string>
#include "descriptor.h"
#include "celllist.h"
#include "geometry.h"

namespace py = pybind11;
using namespace std;

/**
 * Base class for global descriptors.
 */
class DescriptorGlobal : public Descriptor {
    public:
        /**
         * @brief Version of 'create' that automatically extends the system
         * based on PBC and calculates celllist.
         * 
         * @param out 
         * @param positions 
         * @param atomic_numbers 
         * @param cell 
         * @param pbc 
         */
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> cell,
            py::array_t<bool> pbc,
            bool return_descriptor,
            bool return_derivatives
        );

        /**
         * @brief Version of 'create' that automatically calculates celllist.
         * 
         * @param out 
         * @param positions 
         * @param atomic_numbers 
         */
        void create(
            py::array_t<double> out, 
            System &system,
            bool return_descriptor,
            bool return_derivatives
        );

        /**
         * @brief Pure virtual function for calculating the feature vectors.
         * 
         * @param out 
         * @param positions 
         * @param atomic_numbers 
         * @param cell_list 
         */
        virtual void create(
            py::array_t<double> out, 
            System &system,
            CellList cell_list,
            bool return_descriptor,
            bool return_derivatives
        ) = 0;

        /**
        * Calculates the numerical derivates with central finite difference.
        *
        * @param derivatives Numpy output array for the derivatives.
        * @param descriptor Numpy output array for the descriptor.
        * @param positions Atomic positions as [n_atoms, 3] numpy array.
        * @param atomic_numbers Atomic numbers as [n_atoms] numpy array.
        * @param cell Simulation cell as [3, 3] numpy array.
        * @param pbc Simulation cell periodicity as [3] numpy array.
        * @param indices Indices of the atoms for which derivatives are calculated for.
        * @param return_descriptor Determines whether descriptors are calculated or not.
        */
        void derivatives_numerical(
            py::array_t<double> out_d,
            py::array_t<double> out,
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> cell,
            py::array_t<bool> pbc,
            py::array_t<int> indices,
            bool return_descriptor
        );

    protected:
        DescriptorGlobal(bool periodic, string average="", double cutoff=0, string normalization="none");
};

#endif
