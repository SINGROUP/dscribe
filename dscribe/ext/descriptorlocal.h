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

#ifndef DESCRIPTORLOCAL_H
#define DESCRIPTORLOCAL_H

#include <pybind11/numpy.h>
#include <string>
#include "descriptor.h"
#include "celllist.h"

namespace py = pybind11;
using namespace std;

/**
 * Base class for local descriptors.
 */
class DescriptorLocal : public Descriptor {
    public:
        DescriptorLocal(bool periodic, string average="", double cutoff=0, string normalization="none");
        /**
         * Versions of 'create' that automatically extends the system based on
         * PBC and calculate celllist.
         */
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> cell,
            py::array_t<bool> pbc,
            py::array_t<double> centers
        );
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> cell,
            py::array_t<bool> pbc,
            py::array_t<int> centers
        );

        /**
         * Versions of 'create' that automatically calculate
         * celllist.
         */
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> centers
        );
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<int> centers
        );

        /**
         * Pure virtual function for calculating the feature vectors.
         */
        virtual void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> centers,
            CellList cell_list
        ) = 0;
        virtual void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<int> centers,
            CellList cell_list
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
        * @param centers The positions at which the descriptor is evaluated at.
        * @param center_indices Indices of the atoms to which the positions correspond to.
        * @param indices Indices of the atoms for which derivatives are calculated for.
        * @param attach Determines whether centers which correspond to atomic
        *   indices move together with the atoms during calculation.
        * @param return_descriptor Determines whether descriptors are calculated or not.
        */
        void derivatives_numerical(
            py::array_t<double> derivatives, 
            py::array_t<double> descriptor, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> cell,
            py::array_t<bool> pbc,
            py::array_t<double> centers,
            py::array_t<int> center_indices,
            py::array_t<int> indices,
            bool attach,
            bool return_descriptor
        );

};

#endif
