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
#include <functional>
#include <iostream>
#include "mbtr.h"
#include "constants.h"

using namespace std;

inline double weight_unity_k1(const int &atomic_number)
{
    return 1;
}

inline double weight_unity_k2(double distance)
{
    return 1;
}

inline double weight_unity_k3(const int &i, const int &j, const int &k, const vector<vector<double> > &distances)
{
    return 1;
}

inline double weight_exponential_k2(double distance, double scale)
{
    double expValue = exp(-scale*distance);
    return expValue;
}

inline double weight_exponential_k3(const int &i, const int &j, const int &k, const vector<vector<double> > &distances, double scale)
{
    double dist1 = distances[i][j];
    double dist2 = distances[j][k];
    double dist3 = distances[k][i];
    double distTotal = dist1 + dist2 + dist3;
    double expValue = exp(-scale*distTotal);

    return expValue;
}

inline double weight_square_k2(double distance)
{
    double value = 1/(distance*distance);
    return value;
}

inline double weight_smooth_k3(const int &i, const int &j, const int &k, const vector<vector<double> > &distances, double sharpness, double cutoff)
{
    double dist1 = distances[i][j];
    double dist2 = distances[j][k];
    double f_ij = 1 + sharpness* pow((dist1/cutoff), (sharpness+1)) - (sharpness+1)* pow((dist1/cutoff), sharpness);
    double f_jk = 1 + sharpness* pow((dist2/cutoff), (sharpness+1)) - (sharpness+1)* pow((dist2/cutoff), sharpness);

    return f_ij*f_jk;
}

inline double geom_atomic_number(const int &atomic_number)
{
    return (double)atomic_number;
}

inline double geom_distance(double distance)
{
    return distance;
}

inline double geom_inverse_distance(double distance)
{
    double invDist = 1/distance;
    return invDist;
}

inline double geom_cosine(const int &i, const int &j, const int &k, const vector<vector<double> > &distances)
{
    double r_ji = distances[j][i];
    double r_ik = distances[i][k];
    double r_jk = distances[j][k];
    double r_ji_square = r_ji*r_ji;
    double r_ik_square = r_ik*r_ik;
    double r_jk_square = r_jk*r_jk;
    double cosine = 0.5/(r_jk*r_ji) * (r_ji_square+r_jk_square-r_ik_square);

    // Due to numerical reasons the cosine might be slightly under -1 or above 1
    // degrees. E.g. acos is not defined then so we clip the values to prevent
    // NaN:s
    cosine = max(-1.0, min(cosine, 1.0));

    return cosine;
}

inline double geom_angle(const int &i, const int &j, const int &k, const vector<vector<double> > &distances)
{
    double cosine = geom_cosine(i, j, k, distances);
    double angle = acos(cosine) * 180.0 / PI;

    return angle;
}

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
    , k2(k2)
    , k3(k3)
    , normalize_gaussians(normalize_gaussians)
{
    this->set_species(species);
    this->set_normalization(normalization);
    this->set_k1(k1);
}

int MBTR::get_number_of_features() const {
    return get_number_of_k1_features()
      + get_number_of_k2_features()
      + get_number_of_k3_features();
}

int MBTR::get_number_of_k1_features() const {
    int n_features = 0;
    int n_species = this->species.size();

    if (this->k1.size() > 0) {
        int n_k1_grid = this->k1["grid"]["n"].cast<int>();
        int n_k1 = n_species * n_k1_grid;
        n_features += n_k1;
    }

    return n_features;
}

int MBTR::get_number_of_k2_features() const {
    int n_features = 0;
    int n_species = this->species.size();

    if (this->k2.size() > 0) {
        int n_k2_grid = this->k2["grid"]["n"].cast<int>();
        int n_k2 = (n_species * (n_species + 1) / 2) * n_k2_grid;
        n_features += n_k2;
    }

    return n_features;
}

int MBTR::get_number_of_k3_features() const {
    int n_features = 0;
    int n_species = this->species.size();

    if (this->k3.size() > 0) {
        int n_k3_grid = this->k3["grid"]["n"].cast<int>();
        int n_k3 = (n_species * n_species * (n_species + 1) / 2) * n_k3_grid;
        n_features += n_k3;
    }

    return n_features;
}

void MBTR::create(py::array_t<double> &out, System &system, CellList &cell_list)
{
    this->calculate_k1(out, system.atomic_numbers);
    this->calculate_k2(out, system.atomic_numbers, cell_list);
    this->normalize_output(out);
    return;
};


/**
 * @brief Used to normalize part of the given one-dimensional py::array
 * in-place.
 * 
 * @param out The array to normalize
 * @param start Start index, defaults to 0
 * @param end End index, defaults to -1 = end of array
 */
void normalize(py::array_t<double> &out, int start = 0, int end = -1) {
    // Gather magnitude
    auto out_mu = out.mutable_unchecked<1>();
    if (end == -1) {
        end = out.size();
    }
    double norm = 0;
    for (int i = start; i < end; ++i) {
        norm += out_mu[i] * out_mu[i];
    }

    // Divide by L2 norm
    double factor = 1 / sqrt(norm);
    for (int i = start; i < end; ++i) {
        out_mu[i] *= factor;
    }
}

void MBTR::normalize_output(py::array_t<double> &out) {
    if (normalization == "l2_each") {
        int n_k1_features = get_number_of_k1_features();
        int n_k2_features = get_number_of_k2_features();
        int n_k3_features = get_number_of_k3_features();
        int start = 0;
        normalize(out, 0, n_k1_features);
        normalize(out, n_k1_features, n_k1_features+n_k2_features);
        normalize(out, n_k1_features+n_k2_features, n_k1_features+n_k2_features+n_k3_features);
    } else if (normalization == "l2") {
        normalize(out, 0, -1);
    }
}

void MBTR::set_k1(py::dict k1) {
    // Default weighting: unity
    if (k1.size() != 0) {
        if (!k1.contains("weighting")) {
            py::dict weighting;
            weighting["function"] = "unity";
            k1["weighting"] = weighting;
        } else if (!k1["weighting"].contains("function")) {
            k1["weighting"]["function"] = "unity";
        }
    }
    this->k1 = k1;
};
void MBTR::set_k2(py::dict k2) {this->k2 = k2;};
void MBTR::set_k3(py::dict k3) {this->k3 = k3;};
void MBTR::set_normalize_gaussians(bool normalize_gaussians) {
    this->normalize_gaussians = normalize_gaussians;
};
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

py::array_t<int> MBTR::get_species() {return this->species;};
bool MBTR::get_periodic() {return this->periodic;};
map<int, int> MBTR::get_species_index_map() {return this->species_index_map;};
py::dict MBTR::get_k1() {return k1;};
py::dict MBTR::get_k2() {return k2;};
py::dict MBTR::get_k3() {return k3;};
string MBTR::get_normalization() {return normalization;};
bool MBTR::get_normalize_gaussians() {return normalize_gaussians;};

inline vector<double> MBTR::gaussian(double center, double weight, double start, double dx, double sigma, int n) {
    // We first calculate the cumulative distribution function for a normal
    // distribution.
    vector<double> cdf(n+1);
    double x = start;
    for (auto &it : cdf) {
        it = 0.5 * (1.0 + erf((x-center)/(sigma * SQRT2)));
        x += dx;
    }

    // The normal distribution is calculated as a derivative of the cumulative
    // distribution, as with coarse discretization this methods preserves the
    // norm better.
    double normalization = weight * (this->normalize_gaussians
      ? 1
      : sigma * SQRT2PI);
    vector<double> pdf(n);
    int i = 0;
    for (auto &it : pdf) {
        it = normalization * (cdf[i+1]-cdf[i])/dx;
        ++i;
    }

    return pdf;
}

void MBTR::calculate_k1(py::array_t<double> &out, py::array_t<int> &atomic_numbers) {
    // Create mutable and unchecked versions
    auto out_mu = out.mutable_unchecked<1>();
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();

    // Get k1 grid setup
    double sigma = this->k1["grid"]["sigma"].cast<double>();
    double min = this->k1["grid"]["min"].cast<double>();
    double max = this->k1["grid"]["max"].cast<double>();
    int n = this->k1["grid"]["n"].cast<int>();
    double dx = (max - min) / (n - 1);
    double start = min - dx/2;

    // Determine the geometry function to use
    string geom_func_name = this->k1["geometry"]["function"].cast<string>();
    function<double(int)> geom_func;
    if (geom_func_name == "atomic_number") {
        geom_func = geom_atomic_number;
    } else {
        throw invalid_argument("Invalid geometry function.");
    }

    // Determine the weighting function to use
    string weight_func_name = this->k1["weighting"]["function"].cast<string>();
    function<double(int)> weight_func;
    if (weight_func_name == "unity") {
        weight_func = weight_unity_k1;
    } else {
        throw invalid_argument("Invalid geometry function.");
    }

    // Loop through all the atoms in the original, non-extended cell
    int n_atoms = atomic_numbers.size();
    for (int i=0; i < n_atoms; ++i) {
        int i_z = atomic_numbers_u(i);
        double geom = geom_func(i_z);
        double weight = weight_func(i_z);

        // Calculate gaussian
        vector<double> gauss = gaussian(geom, weight, start, dx, sigma, n);

        // Get the index of the present elements in the final vector
        int i_index = this->species_index_map.at(i_z);

        // Sum gaussian into output
        for (int j=0; j < gauss.size(); ++j) {
            out_mu[i_index * n + j] += gauss[j];
        }
    }
}

void MBTR::calculate_k2(py::array_t<double> &out, py::array_t<int> &atomic_numbers, CellList &cell_list) {
    // Create mutable and unchecked versions
    // auto out_mu = out.mutable_unchecked<1>();
    // auto atomic_numbers_u = atomic_numbers.unchecked<1>();

    // // Get k2 grid setup
    // double sigma = this->k2["grid"]["sigma"].cast<double>();
    // double min = this->k2["grid"]["min"].cast<double>();
    // double max = this->k2["grid"]["max"].cast<double>();
    // int n = this->k2["grid"]["n"].cast<int>();
    // double dx = (max - min) / (n - 1);
    // double start = min - dx/2;

    // // Determine the geometry function to use
    // string geom_func_name = this->k2["geometry"]["function"].cast<string>();
    // function<double(double)> geom_func;
    // if (geom_func_name == "distance") {
    //     geom_func = geom_distance;
    // } else if (geom_func_name == "inverse_distance") {
    //     geom_func = geom_inverse_distance;
    // } else {
    //     throw invalid_argument("Invalid geometry function.");
    // }

    // // Determine the weighting function to use
    // string weight_func_name = this->k2["weighting"]["function"].cast<string>();
    // function<double(double)> weight_func;
    // if (weight_func_name == "unity") {
    //     weight_func = weight_unity_k2;
    // } else {
    //     throw invalid_argument("Invalid geometry function.");
    // }

    // // Loop over all atoms in the system
    // int n_atoms = atomic_numbers.size();
    // for (int i=0; i < n_atoms; ++i) {

    //     // For each atom we loop only over the neighbours
    //     CellListResult neighbours = cell_list.getNeighboursForIndex(i);
    //     int n_neighbours = neighbours.indices.size();
    //     for (int i_neighbour = 0; i_neighbour < n_neighbours; ++i_neighbour) {
    //         int j = neighbours.indices[i_neighbour];
    //         double distance = neighbours.distances[i_neighbour];
    //         if (j > i) {

    //             // Only consider pairs that have one atom in the original cell
    //             if (i < this->interaction_limit || j < this->interaction_limit) {
    //                 double geom = geom_func(distance);
    //                 double weight = weight_func(distance);

    //                 // When the pair of atoms are in different copies of the
    //                 // cell, the weight is halved. This is done in order to
    //                 // avoid double counting the same distance in the opposite
    //                 // direction. This correction makes periodic cells with
    //                 // different translations equal and also supercells equal to
    //                 // the primitive cell within a constant that is given by the
    //                 // number of repetitions of the primitive cell in the
    //                 // supercell.
    //                 vector<int> i_copy = this->cell_indices[i];
    //                 vector<int> j_copy = this->cell_indices[j];
    //                 if (i_copy != j_copy) {
    //                     weight /= 2;
    //                 }

    //                 // Calculate gaussian
    //                 vector<double> gauss = gaussian(geom, weight, start, dx, sigma, n);

    //                 // Get the index of the present elements in the final vector
    //                 int i_z = atomic_numbers_u[i];
    //                 int j_z = atomic_numbers_u[j];

    //                 // Get the starting index of the species pair in the final vector
    //                 int i_index = get_location(i_z, j_z)[0];

    //                 // Sum gaussian into output
    //                 for (int j=0; j < gauss.size(); ++j) {
    //                     out_mu[i_index + j] += gauss[j];
    //                 }
    //     }
    // }
}
