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

#include "descriptorglobal.h"

using namespace std;

DescriptorGlobal::DescriptorGlobal(bool periodic, string average, double cutoff)
    : periodic(periodic)
    , average(average)
    , cutoff(cutoff)
{
}

/**
 */
void DescriptorGlobal::derivatives_numerical(
    py::array_t<double> derivatives, 
    py::array_t<double> descriptor, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    py::array_t<int> indices,
    bool attach,
    bool return_descriptor
) const
{
}
