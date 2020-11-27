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

#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H

#include <pybind11/numpy.h>
#include <string>

namespace py = pybind11;
using namespace std;

/**
 * Descriptor base class.
 */
class Descriptor {
    public:
        /**
         * Pure virtual function for creating.
         */
        virtual void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> centers
        ) const = 0; 

        /**
         * Pure virtual function for getting the number of features.
         */
        virtual int get_number_of_features() const = 0; 

        /**
         * For creating derivatives.
         */
        void derivatives_numerical(
            py::array_t<double> out_d, 
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> centers,
            py::array_t<int> indices,
            bool return_descriptor
        ) const;
};

#endif
