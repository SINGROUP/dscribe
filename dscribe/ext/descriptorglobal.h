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
#include "celllist.h"

namespace py = pybind11;
using namespace std;

/**
 * Descriptor base class.
 */
class DescriptorGlobal {
    public:
        /**
         * Nothing precalculated, called by python.
         */
        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> cell,
            py::array_t<bool> pbc
        ) const;

        /**
         * Called internally. The system should already be extended
         * periodically and CellList should be available.
         */
        virtual void create_raw(
            py::detail::unchecked_mutable_reference<double, 1> &out_mu, 
            py::detail::unchecked_reference<double, 2> &positions_u,
            py::detail::unchecked_reference<int, 1> &atomic_numbers_u,
            CellList &cell_list
        ) const = 0; 

        /**
         * Pure virtual function for getting the number of features.
         */
        virtual int get_number_of_features() const = 0; 

        /**
         * Derivatives for global descriptors.
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
        ) const;

    protected:
        DescriptorGlobal(bool periodic, string average="", double cutoff=0);
        const bool periodic;
        const string average;
        const double cutoff;
};

#endif
