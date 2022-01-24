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
#include "celllist.h"

namespace py = pybind11;
using namespace std;

/**
 * Local descriptor base class.
 */
class DescriptorLocal {
    public:
        /**
        * Calculates the feature vector for a periodic system with no
        * precalculated cell list.
        *
        * @param out Numpy output array for the descriptor.
        * @param positions Atomic positions as [n_atoms, 3] numpy array.
        * @param atomic_numbers Atomic numbers as [n_atoms] numpy array.
        * @param centers Positions of the local centers as [n_centers, 3] numpy array.
        * @param cell Simulation cell as [3, 3] numpy array.
        * @param pbc Simulation cell periodicity as [3] numpy array.
        */
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> centers,
            py::array_t<double> cell,
            py::array_t<bool> pbc
        ); 

        /**
        * Calculates the feature vector for a finite system with no
        * precalculated cell list.
        *
        * @param out Numpy output array for the descriptor.
        * @param positions Atomic positions as [n_atoms, 3] numpy array.
        * @param atomic_numbers Atomic numbers as [n_atoms] numpy array.
        * @param centers Positions of the local centers as [n_centers, 3] numpy array.
        * @param cell Simulation cell as [3, 3] numpy array.
        * @param pbc Simulation cell periodicity as [3] numpy array.
        */
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> centers
        );

        /**
        * Calculates the feature vector for a finite system with a
        * precalculated cell list.
        *
        * @param out Numpy output array for the descriptor.
        * @param positions Atomic positions as [n_atoms, 3] numpy array.
        * @param atomic_numbers Atomic numbers as [n_atoms] numpy array.
        * @param centers Positions of the local centers as [n_centers, 3] numpy array.
        * @param cell Simulation cell as [3, 3] numpy array.
        * @param pbc Simulation cell periodicity as [3] numpy array.
        */
        virtual void create(
            py::array_t<double> &out, 
            py::array_t<double> &positions,
            py::array_t<int> &atomic_numbers,
            py::array_t<double> &centers,
            CellList &cellList
        ) = 0; 

        /**
         * Pure virtual function for getting the number of features.
         */
        virtual int get_number_of_features() const = 0; 

        /**
        * Calculates the numerical derivates with central finite difference.
        *
        * @param out_d Numpy output array for the derivatives.
        * @param out Numpy output array for the descriptor.
        * @param positions Atomic positions as [n_atoms, 3] numpy array.
        * @param atomic_numbers Atomic numbers as [n_atoms] numpy array.
        * @param cell Simulation cell as [3, 3] numpy array.
        * @param pbc Simulation cell periodicity as [3] numpy array.
        * @param centers Positions of the local centers as [n_centers, 3] numpy array.
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
            py::array_t<double> centers,
            py::array_t<int> center_indices,
            py::array_t<int> indices,
            bool attach,
            bool return_descriptor
        );

    protected:
        DescriptorLocal(bool periodic, string average="", double cutoff=0);
        const bool periodic;
        const string average;
        const double cutoff;
};

#endif
