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
#include <limits>
#include "descriptor.h"
#include "celllist.h"
#include "geometry.h"

namespace py = pybind11;
using namespace std;

/**
 * Global descriptor base class.
 */
class DescriptorGlobal : public Descriptor {
    protected:
        DescriptorGlobal(bool periodic, string average="", double cutoff=numeric_limits<double>::infinity())
        : Descriptor(periodic, average, cutoff) {};

    public:
        /**
        * Calculates the feature vector for a periodic system with no
        * precalculated cell list.
        *
        * @param out Numpy output array for the descriptor.
        * @param positions Atomic positions as [n_atoms, 3] numpy array.
        * @param atomic_numbers Atomic numbers as [n_atoms] numpy array.
        * @param cell Simulation cell as [3, 3] numpy array.
        * @param pbc Simulation cell periodicity as [3] numpy array.
        */
        void create(py::array_t<double> out, System system); 

        /**
        * Calculates the feature vector for a finite system with a
        * precalculated cell list.
        */
        virtual void create(py::array_t<double> &out, System &system, CellList &cell_list) = 0; 

        /**
         * Pure virtual function for getting the number of features.
         */
        virtual int get_number_of_features() const = 0; 

        /**
        * Calculates the numerical derivates with central finite difference.
        *
        * @param out_d Numpy output array for the derivatives.
        * @param out Numpy output array for the descriptor.
        * @param system The final atomic system.
        * @param indices Indices of the atoms for which derivatives are calculated for.
        * @param return_descriptor Determines whether descriptors are calculated or not.
        */
        void derivatives_numerical(
            py::array_t<double> out_d,
            py::array_t<double> out,
            System &system,
            py::array_t<int> indices,
            bool return_descriptor
        );
};

#endif
