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
#include <limits>
#include <sstream>
#include <unordered_set>
#include "mbtr.h"
#include "constants.h"


using namespace std;

inline double weight_unity_k1(int atomic_number) {
    return 1;
}

inline double weight_unity_k2(double distance) {
    return 1;
}

inline double weight_unity_k3(double distance_ij, double distance_jk, double distance_ki) {
    return 1;
}

inline double weight_exponential_k2(double distance, double scale) {
    double expValue = exp(-scale*distance);
    return expValue;
}

inline double weight_exponential_k3(double distance_ij, double distance_jk, double distance_ki, double scale) {
    double distTotal = distance_ij + distance_jk + distance_ki;
    double expValue = exp(-scale*distTotal);
    return expValue;
}

inline double weight_square_k2(double distance) {
    double value = 1/(distance*distance);
    return value;
}

inline double weight_smooth_k3(double distance_ij, double distance_jk, double distance_ki, double sharpness, double cutoff) {
    double f_ij = 1 + sharpness * pow((distance_ij/cutoff), (sharpness+1)) - (sharpness+1)* pow((distance_ij/cutoff), sharpness);
    double f_jk = 1 + sharpness * pow((distance_jk/cutoff), (sharpness+1)) - (sharpness+1)* pow((distance_jk/cutoff), sharpness);
    return f_ij*f_jk;
}

inline double geom_atomic_number(int atomic_number) {
    return (double)atomic_number;
}

inline double geom_distance(double distance) {
    return distance;
}

inline double geom_inverse_distance(double distance) {
    double invDist = 1/distance;
    return invDist;
}

inline double geom_cosine(double distance_ij, double distance_jk, double distance_ki) {
    double distance_ji_square = distance_ij*distance_ij;
    double distance_ik_square = distance_ki*distance_ki;
    double distance_jk_square = distance_jk*distance_jk;
    double cosine = 0.5/(distance_jk*distance_ij) * (distance_ji_square+distance_jk_square-distance_ik_square);

    // Due to numerical reasons the cosine might be slightly under -1 or above 1
    // degrees. E.g. acos is not defined then so we clip the values to prevent
    // NaN:s
    cosine = max(-1.0, min(cosine, 1.0));
    return cosine;
}

inline double geom_angle(double distance_ij, double distance_jk, double distance_ki) {
    double cosine = geom_cosine(distance_ij, distance_jk, distance_ki);
    double angle = acos(cosine) * 180.0 / PI;
    return angle;
}

inline bool same_cell(const py::array_t<int> &cell_indices, int i, int j) {
    auto cell_indices_u = cell_indices.unchecked<2>();
    for (int k = 0; k < 3; ++k) {
        if (cell_indices_u(i, k) != cell_indices_u(j, k)) {
            return false;
        }
    }
    return true;
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
    , cutoff_k2(numeric_limits<double>::infinity())
    , cutoff_k3(numeric_limits<double>::infinity())
{
    this->set_species(species);
    this->set_normalization(normalization);
    this->set_k1(k1);
    this->set_k2(k2);
    this->set_k3(k3);
    this->set_periodic(periodic);
    this->validate();
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

void MBTR::create(py::array_t<double> &out, System &system, CellList &cell_list) {
    this->calculate_k1(out, system);
    this->calculate_k2(out, system, cell_list);
    this->calculate_k3(out, system, cell_list);
    this->normalize_output(out);
    return;
}

pair<int, int> MBTR::get_location(int z1) {
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


pair<int, int> MBTR::get_location(int z1, int z2) {
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
    if (numbers[0] > numbers[1]) {
        reverse(numbers.begin(), numbers.end());
    }
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

pair<int, int> MBTR::get_location(int z1, int z2, int z3) {
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

    // Reverse if order is not correct
    vector<int> numbers = {i, j, k};
    if (numbers[0] > numbers[2]) {
        reverse(numbers.begin(), numbers.end());
    }
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
    double cutoff = this->get_cutoff(k2);
    this->cutoff_k2 = cutoff;
    this->cutoff = max(this->cutoff_k2, this->cutoff_k3);
}

void MBTR::set_k3(py::dict k3) {
    // Default sharpness = 2
    if (k3.size() != 0 && k3.contains("weighting")) {
        py::dict weighting = k3["weighting"];
        if (weighting.contains("function")) {
            string function = weighting["function"].cast<string>();
            if (function == "smooth_cutoff" && !weighting.contains("sharpness")) {
                weighting["sharpness"] = 2;
            }
        }
    }

    this->k3 = k3;

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
}

double MBTR::get_cutoff(py::dict &k) {
    double cutoff = numeric_limits<double>::infinity();
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
                        if (!weighting.contains("threshold")) {
                            throw invalid_argument("Missing value for 'threshold'.");
                        }
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

void MBTR::validate() {
    this->assert_valle();
    this->assert_weighting(this->k1, 1, unordered_set<string>({"unity"}));
    this->assert_weighting(this->k2, 2, unordered_set<string>({"unity", "exp", "exponential", "inverse_square"}));
    this->assert_weighting(this->k3, 3, unordered_set<string>({"unity", "exp", "exponential", "inverse_square"}));
    this->assert_periodic_weighting(this->k2);
    this->assert_periodic_weighting(this->k3);
    this->assert_geometry(this->k1, 1, unordered_set<string>({"atomic_number"}));
    this->assert_geometry(this->k2, 2, unordered_set<string>({"distance", "inverse_distance"}));
    this->assert_geometry(this->k3, 3, unordered_set<string>({"angle", "cosine"}));
}

void MBTR::assert_valle() {
    if (this->normalization == "valle_oganov" && !this->periodic) {
        throw invalid_argument("Valle-Oganov normalization does not support non-periodic systems.");
    }
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

void MBTR::assert_weighting(py::dict &k, int degree, unordered_set<string> valid_functions) {
    ostringstream os;
    if (k.size() != 0) {
        if (k.contains("weighting")) {
            py::dict weighting = k["weighting"];
            if (weighting.contains("function")) {
                string function = weighting["function"].cast<string>();
                if (valid_functions.find(function) == valid_functions.end()) {
                    os << "Unknown weighting function specified for k" << degree << ".";
                    throw invalid_argument(os.str());
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
                    } else if (function == "smooth_cutoff") {
                        if (!weighting.contains("r_cut")) {
                            throw invalid_argument("Missing value for 'r_cut'.");
                        }
                    }
                }
            }
        }
    }
}

void MBTR::assert_geometry(py::dict &k, int degree, unordered_set<string> valid_functions) {
    ostringstream os;
    if (k.size() != 0) {
        if (k.contains("geometry")) {
            py::dict weighting = k["geometry"];
            if (weighting.contains("function")) {
                string function = weighting["function"].cast<string>();
                if (valid_functions.find(function) == valid_functions.end()) {
                    os << "Unknown geometry function specified for k" << degree << ".";
                    throw invalid_argument(os.str());
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

inline void MBTR::add_gaussian(double center, double weight, double start, double dx, double sigma, int n, const pair<int, int> &loc, py::detail::unchecked_mutable_reference<double, 1> &out_mu) {
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

    int output_start = loc.first;
    for (int i=0; i < n; ++i) {
        out_mu[output_start + i] += normalization * (cdf[i+1]-cdf[i]) / dx;
    }
}

void MBTR::calculate_k1(py::array_t<double> &out, System &system) {
    if (this->k1.size() == 0) {
        return;
    }
    // Create mutable and unchecked versions
    py::array_t<int> atomic_numbers = system.atomic_numbers;
    auto out_mu = out.mutable_unchecked<1>();
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();

    // Get grid setup
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
        throw invalid_argument("Invalid geometry function for k=1.");
    }

    // Determine the weighting function to use
    string weight_func_name = this->k1["weighting"]["function"].cast<string>();
    function<double(int)> weight_func;
    if (weight_func_name == "unity") {
        weight_func = weight_unity_k1;
    } else {
        throw invalid_argument("Invalid weighting function for k=1.");
    }

    // Loop through all the atoms in the original, non-extended cell
    for (auto &i : system.interactive_atoms) {
        int i_z = atomic_numbers_u(i);
        double geom = geom_func(i_z);
        double weight = weight_func(i_z);

        // Get the index of the present elements in the final vector
        pair<int, int> loc = get_location(i_z);

        // Add gaussian to output
        add_gaussian(geom, weight, start, dx, sigma, n, loc, out_mu);
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

    // Get grid setup
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
        throw invalid_argument("Invalid geometry function for k=2.");
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
        throw invalid_argument("Invalid wighting function for k=2.");
    }

    // Loop over all atoms in the system
    int n_atoms = atomic_numbers.size();
    // auto cell_indices_u = system.cell_indices.unchecked<2>();

    // TODO: There may be a more efficent way of looping through the atoms.
    // Maybe looping over the interactive atoms only? Also maybe iterating over
    // the cells only in the positive lattice vector direction?
    for (int i=0; i < n_atoms; ++i) {
        // For each atom we loop only over the neighbours
        unordered_map<int, pair<double, double>> neighbours_i = cell_list.getNeighboursForIndex(i);
        for (auto& it: neighbours_i) {
            int j = it.first;
            double distance = it.second.first;
            if (j > i) {
                // Only consider pairs that have at least one atom in the
                // 'interaction subset', typically the original cell but can
                // also be another local region.
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
                    if (!same_cell(system.cell_indices, i, j)) {
                        weight /= 2;
                    }

                    // Get the starting index of the species pair in the final vector
                    int i_z = atomic_numbers_u(i);
                    int j_z = atomic_numbers_u(j);
                    pair<int, int> loc = get_location(i_z, j_z);

                    // Add gaussian to output
                    add_gaussian(geom, weight, start, dx, sigma, n, loc, out_mu);
                }
            }
        }
    }
}

void MBTR::calculate_k3(py::array_t<double> &out, System &system, CellList &cell_list) {
    if (this->k3.size() == 0) {
        return;
    }
    // Create mutable and unchecked versions
    auto out_mu = out.mutable_unchecked<1>();
    auto atomic_numbers = system.atomic_numbers;
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();

    // Get grid setup
    double sigma = this->k3["grid"]["sigma"].cast<double>();
    double min = this->k3["grid"]["min"].cast<double>();
    double max = this->k3["grid"]["max"].cast<double>();
    int n = this->k3["grid"]["n"].cast<int>();
    double dx = (max - min) / (n - 1);
    double start = min - dx/2;

    // Determine the geometry function to use
    string geom_func_name = this->k3["geometry"]["function"].cast<string>();
    function<double(double, double, double)> geom_func;
    if (geom_func_name == "angle") {
        geom_func = geom_angle;
    } else if (geom_func_name == "cosine") {
        geom_func = geom_cosine;
    } else {
        throw invalid_argument("Invalid geometry function for k=3.");
    }

    // Determine the weighting function to use
    string weight_func_name = this->k3["weighting"]["function"].cast<string>();
    function<double(double, double, double)> weight_func;
    if (weight_func_name == "unity") {
        weight_func = weight_unity_k3;
    } else if (weight_func_name == "exp" || weight_func_name == "exponential") {
        double scale = this->k3["weighting"]["scale"].cast<double>();
        weight_func = bind(
            weight_exponential_k3,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            scale
        );
    } else if (weight_func_name == "smooth_cutoff") {
        double sharpness = this->k3["weighting"]["sharpness"].cast<double>();
        double r_cut = this->k3["weighting"]["r_cut"].cast<double>();
        weight_func = bind(
            weight_smooth_k3,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            sharpness,
            r_cut
        );
    } else {
        throw invalid_argument("Invalid weighting function for k=3.");
    }

    // For each atom we loop only over the atoms triplets that are within the
    // neighbourhood
    int n_atoms = atomic_numbers.size();
    auto cell_indices = system.cell_indices;
    for (int i=0; i < n_atoms; ++i) {
        unordered_map<int, pair<double, double>> neighbours_i = cell_list.getNeighboursForIndex(i);
        for (auto& it_i: neighbours_i) {
            int j = it_i.first;
            unordered_map<int, pair<double, double>> neighbours_j = cell_list.getNeighboursForIndex(j);
            for (auto& it_j: neighbours_j) {
                int k = it_j.first;
                // Only consider triples that have at least one atom in the
                // 'interaction subset', typically the original cell but can
                // also be another local region.
                bool i_interactive = system.interactive_atoms.find(i) != system.interactive_atoms.end();
                bool j_interactive = system.interactive_atoms.find(j) != system.interactive_atoms.end();
                bool k_interactive = system.interactive_atoms.find(k) != system.interactive_atoms.end();
                if (i_interactive || j_interactive || k_interactive) {
                    // Calculate angle for all index permutations from choosing
                    // three out of n_atoms. The same atom cannot be present
                    // twice in the permutation.
                    if (j != i && k != j && k != i) {
                        // The angles are symmetric: ijk = kji. The value is
                        // calculated only for the triplet where k > i.
                        if (k > i) {
                            double distance_ij = neighbours_i[j].first;
                            double distance_jk = neighbours_j[k].first;
                            double distance_ki = neighbours_i[k].first;

                            // Calculate geometry value
                            double geom = geom_func(distance_ij, distance_jk, distance_ki);

                            // Calculate weight value
                            double weight = weight_func(distance_ij, distance_jk, distance_ki);

                            // The contributions are weighted by their multiplicity arising from
                            // the translational symmetry. Each triple of atoms is repeated N
                            // times in the extended system through translational symmetry. The
                            // weight for the angles is thus divided by N so that the
                            // multiplication from symmetry is countered. This makes the final
                            // spectrum invariant to the selected supercell size and shape
                            // after normalization. The number of repetitions N is given by how
                            // many unique cell indices (the index of the repeated cell with
                            // respect to the original cell at index [0, 0, 0]) are present for
                            // the atoms in the triple.
                            int diff_sum =
                                (int)!same_cell(cell_indices, i, j)
                              + (int)!same_cell(cell_indices, i, k)
                              + (int)!same_cell(cell_indices, j, k);
                            if (diff_sum > 1) {
                                weight /= diff_sum;
                            }

                            // Get the starting index of the species triple in
                            // the final vector
                            int i_z = atomic_numbers_u(i);
                            int j_z = atomic_numbers_u(j);
                            int k_z = atomic_numbers_u(k);
                            pair<int, int> loc = get_location(i_z, j_z, k_z);

                            // Add gaussian to output
                            add_gaussian(geom, weight, start, dx, sigma, n, loc, out_mu);
                        }
                    }
                }
            }
        }
    }
}
