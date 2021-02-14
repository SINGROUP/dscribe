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
#include "dok.h"

using namespace std;

DOK::DOK()
{
}

double& DOK::operator()(int i, int j, int k, int l)
{
    return this->container[make_tuple(i, j, k, l)];
}

py::array_t<double> DOK::data()
{
    int n_items = this->container.size();
    py::array_t<double> data({n_items});
    auto data_mu = data.mutable_unchecked<1>();

    int i = 0;
    for (auto &kv : this->container) {
        data_mu(i) = kv.second;
        ++i;
    }
    return data;
}
py::array_t<int> DOK::coords()
{
    int n_items = this->container.size();
    py::array_t<int> coords({4, n_items});
    auto coords_mu = coords.mutable_unchecked<2>();

    int i = 0;
    for (auto &kv : this->container) {
        coords_mu(0, i) = get<0>(kv.first);
        coords_mu(1, i) = get<1>(kv.first);
        coords_mu(2, i) = get<2>(kv.first);
        coords_mu(3, i) = get<3>(kv.first);
        ++i;
    }
    return coords;
}
