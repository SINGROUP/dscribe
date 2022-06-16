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
#include <algorithm>
#include <iostream>
#include <unordered_set>
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
    , normalize_gaussians(normalize_gaussians)
    , cutoff_k2(0)
    , cutoff_k3(0)
{
    this->set_species(species);
    this->set_normalization(normalization);
    this->set_k1(k1);
    this->set_k2(k2);
    this->set_k3(k3);
    this->set_periodic(periodic);
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
    this->calculate_k1(out, system);
    this->calculate_k2(out, system, cell_list);
    this->normalize_output(out);
    return;
}

pair<int, int> MBTR::get_location(int z1)
{
    // Check that the corresponding part is calculated
    if (this->k1.size() == 0) {
        throw invalid_argument(
            "Cannot retrieve the location for {}, as the term k1 has not "
            "been specified."
        );
    }

    // Change into internal indexing
    int m = this->species_index_map[z1];

    // Get the start and end index
    int n = this->k1["grid"]["n"].cast<int>();
    int start = m * n;
    int end = (m + 1) * n;

    return make_pair(start, end);
};


pair<int, int> MBTR::get_location(int z1, int z2)
{
    // Check that the corresponding part is calculated
    if (this->k2.size() == 0) {
        throw invalid_argument(
            "Cannot retrieve the location for {}, as the term k2 has not "
            "been specified."
        );
    }

    // Change into internal indexing
    int i = this->species_index_map[z1];
    int j = this->species_index_map[z2];

    // Sort
    vector<int> numbers = {i, j};
    sort(numbers.begin(), numbers.end());
    i = numbers[0];
    j = numbers[1];

    // This is the index of the spectrum. It is given by enumerating the
    // elements of an upper triangular matrix from left to right and top
    // to bottom.
    int n = this->k2["grid"]["n"].cast<int>();
    int n_elem = this->species.size();
    int m = j + i * n_elem - i * (i + 1) / 2;
    int offset = get_number_of_k1_features();
    int start = offset + m * n;
    int end = offset + (m + 1) * n;

    return make_pair(start, end);
};

pair<int, int> MBTR::get_location(int z1, int z2, int z3)
{
    // Check that the corresponding part is calculated
    if (this->k3.size() == 0) {
        throw invalid_argument(
            "Cannot retrieve the location for {}, as the term k3 has not "
            "been specified."
        );
    }

    // Change into internal indexing
    int i = this->species_index_map[z1];
    int j = this->species_index_map[z2];
    int k = this->species_index_map[z3];

    // Sort
    vector<int> numbers = {i, j, k};
    sort(numbers.begin(), numbers.end());
    i = numbers[0];
    j = numbers[1];
    k = numbers[2];

    // This is the index of the spectrum. It is given by enumerating the
    // elements of a three-dimensional array where for valid elements
    // k>=i. The enumeration begins from [0, 0, 0], and ends at [n_elem,
    // n_elem, n_elem], looping the elements in the order k, i, j.
    int n = this->k3["grid"]["n"].cast<int>();
    int n_elem = this->species.size();
    int m = j * n_elem * (n_elem + 1) / 2 + k + i * n_elem - i * (i + 1) / 2;

    int offset = get_number_of_k1_features() + get_number_of_k2_features();
    int start = offset + m * n;
    int end = offset + (m + 1) * n;

    return make_pair(start, end);
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
}

void MBTR::set_k2(py::dict k2) {
    this->k2 = k2;
    this->assert_periodic_weighting(k2);
    this->assert_weighting(k2);
    double cutoff = this->get_cutoff(k2);
    this->cutoff_k2 = cutoff;
    this->cutoff = max(this->cutoff_k2, this->cutoff_k3);
}

void MBTR::set_k3(py::dict k3) {
    this->k3 = k3;
    this->assert_periodic_weighting(k3);
    this->assert_weighting(k3);
    // In k3, the distance is defined as the perimeter, thus we half the
    // distance to get the actual cutoff.
    double cutoff = 0.5 * this->get_cutoff(k3);
    this->cutoff_k3 = cutoff;
    this->cutoff = max(this->cutoff_k2, this->cutoff_k3);
}

void MBTR::set_normalize_gaussians(bool normalize_gaussians) {
    this->normalize_gaussians = normalize_gaussians;
}

void MBTR::set_normalization(string normalization) {
    this->normalization = normalization;
    assert_valle();
}

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
}

void MBTR::set_periodic(bool periodic) {
    this->periodic = periodic;
    assert_valle();
    assert_periodic_weighting(this->k2);
    assert_periodic_weighting(this->k3);
}

void MBTR::assert_valle() {
    if (this->normalization == "valle_oganov" && !this->periodic) {
        throw invalid_argument("Valle-Oganov normalization does not support non-periodic systems.");
    }
}

double MBTR::get_cutoff(py::dict &k) {
    double cutoff = 0;
    if (k.size() != 0) {
        if (k.contains("weighting")) {
            py::dict weighting = k["weighting"];
            if (weighting.contains("function")) {
                string function = weighting["function"].cast<string>();
                if (function == "exp" || function == "exponential") {
                    if (weighting.contains("r_cut")) {
                        cutoff = weighting["r_cut"].cast<double>();
                    } else if (weighting.contains("scale")) {
                        double scale = weighting["scale"].cast<double>();
                        double threshold = weighting["threshold"].cast<double>();
                        cutoff = -log(threshold) / scale;
                    }
                } else if (function == "inverse_square") {
                    if (weighting.contains("r_cut")) {
                        cutoff = weighting["r_cut"].cast<double>();
                    }
                }
            }
        }
    }
    return cutoff;
}

void MBTR::assert_periodic_weighting(py::dict &k) {
    if (this->periodic) {
        if (k.size() != 0) {
            bool valid = false;
            if (k.contains("weighting")) {
                py::dict weighting = k["weighting"];
                if (weighting.contains("function")) {
                    string function = weighting["function"].cast<string>();
                    if (function != "unity") {
                        valid = true;
                    }
                }
            }
            if (!valid) {
                throw invalid_argument("Periodic systems need to have a weighting function.");
            }
        }
    }
}

void MBTR::assert_weighting(py::dict &k) {
    if (k.size() != 0) {
        if (k.contains("weighting")) {
            py::dict weighting = k["weighting"];
            if (weighting.contains("function")) {
                string function = weighting["function"].cast<string>();
                unordered_set<string> valid_functions( {"unity", "exp", "exponential", "inverse_square"} );
                if (valid_functions.find(function) == valid_functions.end()) {
                    throw invalid_argument("Unknown weighting function specified.");
                } else {
                    if (function == "exp" || function == "exponential") {
                        if (!weighting.contains("threshold")) {
                            throw invalid_argument("Missing value for 'threshold'.");
                        }
                        if (!weighting.contains("scale") && !weighting.contains("r_cut")) {
                            throw invalid_argument("Provide either 'scale' or 'r_cut'.");
                        }
                    } else if (function == "inverse_square") {
                        if (!weighting.contains("r_cut")) {
                            throw invalid_argument("Missing value for 'r_cut'.");
                        }
                    }
                }
            }
        }
    }
}

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

void MBTR::calculate_k1(py::array_t<double> &out, System &system) {
    if (this->k1.size() == 0) {
        return;
    }
    // Create mutable and unchecked versions
    py::array_t<int> atomic_numbers = system.atomic_numbers;
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
    for (auto &i : system.interactive_atoms) {
        int i_z = atomic_numbers_u(i);
        double geom = geom_func(i_z);
        double weight = weight_func(i_z);

        // Calculate gaussian
        vector<double> gauss = gaussian(geom, weight, start, dx, sigma, n);

        // Get the index of the present elements in the final vector
        pair<int, int> location = get_location(i_z);
        int start = location.first;

        // Sum gaussian into output
        for (int j=0; j < gauss.size(); ++j) {
            out_mu[start + j] += gauss[j];
        }
    }
}

void MBTR::calculate_k2(py::array_t<double> &out, System &system, CellList &cell_list) {
    if (this->k2.size() == 0) {
        return;
    }
    // Create mutable and unchecked versions
    auto out_mu = out.mutable_unchecked<1>();
    auto atomic_numbers = system.atomic_numbers;
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();

    // Get k2 grid setup
    double sigma = this->k2["grid"]["sigma"].cast<double>();
    double min = this->k2["grid"]["min"].cast<double>();
    double max = this->k2["grid"]["max"].cast<double>();
    int n = this->k2["grid"]["n"].cast<int>();
    double dx = (max - min) / (n - 1);
    double start = min - dx/2;

    // Determine the geometry function to use
    string geom_func_name = this->k2["geometry"]["function"].cast<string>();
    function<double(double)> geom_func;
    if (geom_func_name == "distance") {
        geom_func = geom_distance;
    } else if (geom_func_name == "inverse_distance") {
        geom_func = geom_inverse_distance;
    } else {
        throw invalid_argument("Invalid geometry function.");
    }

    // Determine the weighting function to use
    string weight_func_name = this->k2["weighting"]["function"].cast<string>();
    function<double(double)> weight_func;
    if (weight_func_name == "unity") {
        weight_func = weight_unity_k2;
    } else if (weight_func_name == "exp" || weight_func_name == "exponential") {
        double scale = this->k2["weighting"]["scale"].cast<double>();
        weight_func = bind(weight_exponential_k2, std::placeholders::_1, scale);
    } else if (weight_func_name == "inverse_square") {
        weight_func = weight_square_k2;
    } else {
        throw invalid_argument("Invalid geometry function.");
    }

    // Loop over all atoms in the system
    int n_atoms = atomic_numbers.size();
    auto cell_indices_u = system.cell_indices.unchecked<2>();
    for (int i=0; i < n_atoms; ++i) {
        // For each atom we loop only over the neighbours
        CellListResult neighbours = cell_list.getNeighboursForIndex(i);

        int n_neighbours = neighbours.indices.size();
        for (int i_neighbour = 0; i_neighbour < n_neighbours; ++i_neighbour) {
            int j = neighbours.indices[i_neighbour];
            double distance = neighbours.distances[i_neighbour];
            if (j > i) {
                // Only consider pairs that have one atom in the 'interaction
                // subset', typically the original cell but can also be another
                // local region.
                bool i_interactive = system.interactive_atoms.find(i) != system.interactive_atoms.end();
                bool j_interactive = system.interactive_atoms.find(j) != system.interactive_atoms.end();
                if (i_interactive || j_interactive) {
                    double geom = geom_func(distance);
                    double weight = weight_func(distance);

                    // When the pair of atoms are in different copies of the
                    // cell, the weight is halved. This is done in order to
                    // avoid double counting the same distance in the opposite
                    // direction. This correction makes periodic cells with
                    // different translations equal and also supercells equal to
                    // the primitive cell within a constant that is given by the
                    // number of repetitions of the primitive cell in the
                    // supercell.
                    bool same_cell = true;
                    for (int k = 0; k < 3; ++k) {
                        if (cell_indices_u(i, k) != cell_indices_u(j, k)) {
                            same_cell = false;
                            break;
                        }
                    }
                    if (!same_cell) {
                        weight /= 2;
                    }

                    // Calculate gaussian
                    vector<double> gauss = gaussian(geom, weight, start, dx, sigma, n);

                    // Get the index of the present elements in the final vector
                    int i_z = atomic_numbers_u(i);
                    int j_z = atomic_numbers_u(j);

                    // Get the starting index of the species pair in the final vector
                    pair<int, int> loc = get_location(i_z, j_z);
                    int start = loc.first;

                    // Sum gaussian into output
                    for (int j=0; j < gauss.size(); ++j) {
                        out_mu[start + j] += gauss[j];
                    }
                }
            }
        }
    }
}
