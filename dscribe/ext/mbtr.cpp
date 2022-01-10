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
#include "mbtr.h"

using namespace std;

MBTR::MBTR(
    const py::dict k1,
    const py::dict k2,
    const py::dict k3,
    bool normalize_gaussians,
    string normalization,
    py::array_t<int> species,
    bool periodic
)
    : DescriptorGlobal(periodic)
    , k1(k1)
    , k2(k2)
    , k3(k3)
    , normalize_gaussians(normalize_gaussians)
    , normalization(normalization)
    , species(species)
{
}

int MBTR::get_number_of_features() const {
    int n_features = 0;
    int n_species = this->species.size();

    if (this->k1.size() > 0) {
        int n_k1_grid = this->k1["grid"]["n"].cast<int>();
        int n_k1 = n_species * n_k1_grid;
        n_features += n_k1;
    }
    if (this->k2.size() > 0) {
        int n_k2_grid = this->k2["grid"]["n"].cast<int>();
        int n_k2 = (n_species * (n_species + 1) / 2) * n_k2_grid;
        n_features += n_k2;
    }
    if (this->k3.size() > 0) {
        int n_k3_grid = this->k3["grid"]["n"].cast<int>();
        int n_k3 = (n_species * n_species * (n_species + 1) / 2) * n_k3_grid;
        n_features += n_k3;
    }

    return n_features;
}

// void MBTR::get_k1(py::detail::unchecked_mutable_reference<double, 1> &out_mu, py::detail::unchecked_reference<int, 1> &atomic_numbers_u) {
//     int n_atoms = Z.size();
//     float dx = (max-min)/(n-1);
//     float sigmasqrt2 = sigma*sqrt(2.0);
//     float start = min-dx/2;

//     // Determine the geometry function to use
//     string geom_func_name = ...;
//     if (geom_func_name == "atomic_number") {
//         geom_func = k1GeomAtomicNumber(i, Z);
//     } else {
//         throw invalid_argument("Invalid geometry function.");
//     }

//     // Determine the weighting function to use
//     string weight_func_name = ...;
//     if (weight_func_name == "atomic_number") {
//         weight_func = k1GeomAtomicNumber(i, Z);
//     } else {
//         throw invalid_argument("Invalid geometry function.");
//     }

//     // Loop through all the atoms in the original, non-extended cell
//     for (int i=0; i < nAtoms; ++i) {
//         float geom = geom_func(i, Z);
//         float weight = weight_func(i, Z);

//         // Calculate gaussian
//         vector<double> gauss = gaussian(geom, weight, start, dx, sigmasqrt2, n);

//         // Get the index of the present elements in the final vector
//         int i_elem = Z[i];
//         int i_index = this->atomicNumberToIndexMap.at(i_elem);

//         // Sum gaussian into output
//         transform(old.begin(), old.end(), gauss.begin(), old.begin(), plus<double>());
//     }
// }

void MBTR::create_raw(
    py::detail::unchecked_mutable_reference<double, 1> &out_mu,
    py::detail::unchecked_reference<double, 2> &positions_u,
    py::detail::unchecked_reference<int, 1> &atomic_numbers_u,
    CellList &cell_list
) {
    return;
};

void MBTR::set_k1(py::dict k1) {this->k1 = k1;};
void MBTR::set_k2(py::dict k2) {this->k2 = k2;};
void MBTR::set_k3(py::dict k3) {this->k3 = k3;};
void MBTR::set_normalize_gaussians(bool normalize_gaussians) {};
void MBTR::set_normalization(string normalization) {
    this->normalization = normalization;
    assert_valle();
};
void MBTR::set_species(py::array_t<int> species) {this->species = species;};
void MBTR::set_periodic(bool periodic) {
    this->periodic = periodic;
    assert_valle();
};
void MBTR::assert_valle() {
    if (this->normalization == "valle_oganov" && !this->periodic) {
        throw std::invalid_argument("Valle-Oganov normalization does not support non-periodic systems.");
    };
};

py::dict MBTR::get_k1() {return k1;};
py::dict MBTR::get_k2() {return k2;};
py::dict MBTR::get_k3() {return k3;};
py::array_t<int> MBTR::get_species() {return this->species;};