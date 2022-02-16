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
{
    this->set_species(species);
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

void MBTR::create(
    py::array_t<double> &out,
    py::array_t<double> &positions,
    py::array_t<int> &atomic_numbers,
    CellList &cell_list
) {
    this->calculate_k1(out, atomic_numbers);
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
void MBTR::set_species(py::array_t<int> species) {
    this->species = species;
    // Setup mappings between atom indices and types together with some
    // statistics
    map<int, int> species_index_map;
    map<int, int> index_species_map;
    auto species_u = species.unchecked<1>();
    int max_atomic_number = species_u(0);
    int min_atomic_number = species_u(0);
    int n_elements = species_u.size();
    for (int i = 0; i < n_elements; ++i) {
        int atomic_number = species_u(i);
        if (atomic_number > max_atomic_number) {
            max_atomic_number = atomic_number;
        }
        if (atomic_number < min_atomic_number) {
            min_atomic_number = atomic_number;
        }
        species_index_map[atomic_number] = i;
        index_species_map[i] = atomic_number;
    }
    this->species_index_map = species_index_map;
    this->index_species_map = index_species_map;
};
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
map<int, int> MBTR::get_species_index_map() {return this->species_index_map;};


inline vector<double> MBTR::gaussian(double center, double weight, double start, double dx, double sigmasqrt2, int n) {
    // We first calculate the cumulative distribution function for a normal
    // distribution.
    vector<double> cdf(n+1);
    double x = start;
    for (auto &it : cdf) {
        it = weight*1.0/2.0*(1.0 + erf((x-center)/sigmasqrt2));
        x += dx;
    }

    // The normal distribution is calculated as a derivative of the cumulative
    // distribution, as with coarse discretization this methods preserves the
    // norm better.
    vector<double> pdf(n);
    int i = 0;
    for (auto &it : pdf) {
        it = (cdf[i+1]-cdf[i])/dx;
        ++i;
    }

    return pdf;
}

void MBTR::calculate_k1(py::array_t<double> &out, py::array_t<int> &atomic_numbers) {
    // Create mutable and unchecked versions
    auto out_mu = out.mutable_unchecked<1>();
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();

    double sigma = this->k1["grid"]["sigma"].cast<double>();
    double min = this->k1["grid"]["min"].cast<double>();
    double max = this->k1["grid"]["max"].cast<double>();
    int n = this->k1["grid"]["n"].cast<int>();
    double dx = (max - min) / (n - 1);
    double sigmasqrt2 = sigma * sqrt(2.0);
    double start = min - dx/2;

    // // Determine the geometry function to use
    // string geom_func_name = ...;
    // if (geom_func_name == "atomic_number") {
    //     geom_func = k1GeomAtomicNumber(i, Z);
    // } else {
    //     throw invalid_argument("Invalid geometry function.");
    // }

    // // Determine the weighting function to use
    // string weight_func_name = ...;
    // if (weight_func_name == "atomic_number") {
    //     weight_func = k1GeomAtomicNumber(i, Z);
    // } else {
    //     throw invalid_argument("Invalid geometry function.");
    // }

    // Loop through all the atoms in the original, non-extended cell
    int n_atoms = atomic_numbers.size();
    for (int i=0; i < n_atoms; ++i) {
        int i_z = atomic_numbers_u(i);
        // double geom = geom_func(i, i_z);
        // double weight = weight_func(i, i_z);

        // Calculate gaussian
        // vector<double> gauss = gaussian(geom, weight, start, dx, sigmasqrt2, n);
        vector<double> gauss = gaussian(1, 1, start, dx, sigmasqrt2, n);

        // Get the index of the present elements in the final vector
        int i_index = this->species_index_map.at(i_z);

        // Sum gaussian into output
        for (int j=0; j < gauss.size(); ++j) {
            out_mu[i_index * n + j] += gauss[j];
        }
    }
}
