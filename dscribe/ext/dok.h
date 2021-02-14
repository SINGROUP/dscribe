
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

#ifndef DOK_H
#define DOK_H

#include <pybind11/numpy.h>
#include <unordered_map>
#include <tuple>

namespace py = pybind11;
using namespace std;

/**
 * Custom hash function for tuples.
 */
struct TupleHasher {
    size_t operator()(const tuple<int, int, int, int>& a) const {
        size_t h = 0;
        h ^= hash<int>{}(get<0>(a)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        h ^= hash<int>{}(get<1>(a)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        h ^= hash<int>{}(get<2>(a)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        h ^= hash<int>{}(get<3>(a)) + 0x9e3779b9 + (h << 6) + (h >> 2); 
        return h;
    }   
};

/**
 * Dictionary Of Keys style sparse array. Has constant time access for reads
 * and writes.
 */
class DOK {
    public:
        /**
         * Constructor
         */
        DOK();
        /**
         * For reserving space.
         */
        void reserve(int n) {this->container.reserve(n);};
        /**
         * For assigning values with parentheses.
         */
        double& operator()(int i, int j, int k, int l);
        /**
         * Data and coords for converting to COO format. A separate function is
         * used for each as this way pybind11 can correctly transfer the
         * ownership of the numpy data to python (compared to using a single
         * function returning a tuple containing both.)
         */
        py::array_t<int> coords();
        py::array_t<double> data();
        /**
         * Data container
         */
        unordered_map<tuple<int, int, int, int>, double, TupleHasher> container;
};

#endif
