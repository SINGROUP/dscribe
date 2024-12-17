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
#include <iostream>
#include <string>
#include <sstream>
#include <set>
#include "mbtr.h"
#include "constants.h"
#include "geometry.h"


using namespace std;

inline double MBTR::get_scale() {
    double scale;
    if (this->weighting.contains("scale")) {
        scale = this->weighting["scale"].cast<double>();
    } else {
        double threshold = this->weighting["threshold"].cast<double>();
        double r_cut = this->weighting["r_cut"].cast<double>();
        scale = - log(threshold) / r_cut;
    }
    return scale;
}

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
    double distance_ij_square = distance_ij*distance_ij;
    double distance_jk_square = distance_jk*distance_jk;
    double distance_ki_square = distance_ki*distance_ki;
    double cosine = 0.5/(distance_ij*distance_jk) * (distance_ij_square+distance_jk_square-distance_ki_square);

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

MBTR::MBTR(
    const py::dict geometry,
    const py::dict grid,
    const py::dict weighting,
    bool normalize_gaussians,
    string normalization,
    py::array_t<int> species,
    bool periodic
)
    : DescriptorGlobal(periodic, "", 0, normalization)
    , normalize_gaussians(normalize_gaussians)
{
    this->set_species(species);
    this->set_normalization(normalization);
    this->set_geometry(geometry);
    this->set_grid(grid);
    this->set_weighting(weighting);
    this->set_periodic(periodic);
    this->validate();
}

int MBTR::get_number_of_features() const {
    int n_species = this->species.size();
    int n_features = 0;
    int n_grid = this->grid["n"].cast<int>();

    if (this->k == 1) {
        n_features = n_species * n_grid;
    } else if (this->k == 2) {
        n_features = (n_species * (n_species + 1) / 2) * n_grid;
    } else if (this->k == 3) {
        n_features = (n_species * n_species * (n_species + 1) / 2) * n_grid;
    }
    return n_features;
}

/**
 * @brief Used to normalize part of the given one-dimensional py::array
 * in-place.
 * 
 * @param out The array to normalize
 * @param start Start index, defaults to 0
 * @param end End index, defaults to -1 = end of array
 */
void MBTR::normalize_output(py::array_t<double> &out, System &system) {
    double factor = 1;
    int end = out.size();
    auto out_mu = out.mutable_unchecked<1>();
    if (this->normalization == "l2") {
        double norm = 0;
        for (int i = 0; i < end; ++i) {
            norm += out_mu[i] * out_mu[i];
        }
        factor = 1 / sqrt(norm);
        for (int i = 0; i < end; ++i) {
            out_mu[i] *= factor;
        }
    } else if (this->normalization == "n_atoms") {
        factor = 1 / system.atomic_numbers.size();
        for (int i = 0; i < end; ++i) {
            out_mu[i] *= factor;
        }
    } else if (this->normalization == "valle_oganov") {
        double volume = get_volume(system.cell);
        std::unordered_map<int, int> counts = count_unique(system.atomic_numbers);
        if (this->k == 2) {
            for (auto& it_i: counts) {
                for (auto& it_j: counts) {
                    int Z_i = it_i.first;
                    int Z_j = it_j.first;
                    int i_z = this->species_index_map[Z_i];
                    int j_z = this->species_index_map[Z_j];
                    if (j_z < i_z) {
                        continue;
                    }
                    double count_product = (Z_i == Z_j)
                        ? 0.5 * counts.at(i_z) * counts.at(j_z)
                        : counts.at(i_z) * counts.at(j_z);
                    double factor = volume / (4.0 * PI * count_product);
                    pair<int, int> location = get_location(i_z, j_z);
                    for (int i = location.first; i < location.second; ++i) {
                        out_mu[i] *= factor;
                    }
                }
            }
        } else if (this->k == 3) {
            for (auto& it_i: counts) {
                for (auto& it_j: counts) {
                    for (auto& it_k: counts) {
                        int Z_i = it_i.first;
                        int Z_j = it_j.first;
                        int Z_k = it_k.first;
                        int i_z = this->species_index_map[Z_i];
                        int j_z = this->species_index_map[Z_j];
                        int k_z = this->species_index_map[Z_k];
                        if (k_z < i_z) {
                            continue;
                        }
                        double count_product = counts.at(i_z) * counts.at(j_z) * counts.at(k_z);
                        double factor = volume /  (4.0 * PI * count_product);
                        pair<int, int> location = get_location(i_z, j_z, k_z);
                        for (int i = location.first; i < location.second; ++i) {
                            out_mu[i] *= factor;
                        }
                    }
                }
            }
        }
    }
}

void MBTR::create(
    py::array_t<double> out, 
    System &system,
    CellList cell_list,
    bool return_descriptor,
    bool return_derivatives
) {
    if (this->k == 1) {
      this->calculate_k1(out, system);
    } else if (this->k == 2) {
        this->calculate_k2(out, system, cell_list);
    } else if (this->k == 3) {
        this->calculate_k3(out, system, cell_list);
    }
    this->normalize_output(out, system);
    return;
}

pair<int, int> MBTR::get_location(int z1) {
    // Check that the corresponding part is calculated
    if (this->k != 1) {
        throw invalid_argument(
            "Cannot retrieve the location for {}, as the term k1 has not "
            "been specified."
        );
    }

    // Change into internal indexing
    int m = this->species_index_map[z1];

    // Get the start and end index
    int n = this->grid["n"].cast<int>();
    int start = m * n;
    int end = (m + 1) * n;

    return make_pair(start, end);
};


pair<int, int> MBTR::get_location(int z1, int z2) {
    // Check that the corresponding part is calculated
    if (this->k != 2) {
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
    int n = this->grid["n"].cast<int>();
    int n_elem = this->species.size();
    int m = j + i * n_elem - i * (i + 1) / 2;
    int start = m * n;
    int end = (m + 1) * n;

    return make_pair(start, end);
};

pair<int, int> MBTR::get_location(int z1, int z2, int z3) {
    // Check that the corresponding part is calculated
    if (this->k != 3) {
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
    int n = this->grid["n"].cast<int>();
    int n_elem = this->species.size();
    int m = j * n_elem * (n_elem + 1) / 2 + k + i * n_elem - i * (i + 1) / 2;

    int start = m * n;
    int end = (m + 1) * n;

    return make_pair(start, end);
};

void MBTR::set_geometry(py::dict geometry) {
    this->geometry = geometry;
    if (geometry.contains("function")) {
        string function = geometry["function"].cast<string>();
        set<string> k1({"atomic_number"});
        set<string> k2({"distance", "inverse_distance"});
        set<string> k3({"angle", "cosine"});
        if (k1.find(function) != k1.end()) {
            this->k = 1;
        } else if (k2.find(function) != k2.end()) {
            this->k = 2;
        } else if (k3.find(function) != k3.end()) {
            this->k = 3;
        } else {
            set<string> valid_functions = k1;
            valid_functions.insert(k2.begin(), k2.end());
            valid_functions.insert(k3.begin(), k3.end());
            std::ostringstream oss;
            oss << "Unknown geometry function. Please use one of the following: ";
            bool first = true;
            for (auto &it : valid_functions) {
                if (!first) {
                    oss << ", ";
                }
                oss << "'" << it << "'";
                first = false;
            }
            oss << ".";
            throw invalid_argument(oss.str());
        }
    } else {
        throw invalid_argument("Please specify a geometry function.");
    }
}

void MBTR::set_grid(py::dict grid) {
    this->grid = grid;
}

void MBTR::set_weighting(py::dict weighting) {
    // Default weighting unity
    if (!weighting.contains("function")) {
        weighting["function"] = "unity";
    }

    // Default sharpness=2 for smooth cutoff
    string function = weighting["function"].cast<string>();
    if (function == "smooth_cutoff" && !weighting.contains("sharpness")) {
        weighting["sharpness"] = 2;
    }

    this->weighting = weighting;
    this->cutoff = this->get_cutoff();
}

void MBTR::set_normalize_gaussians(bool normalize_gaussians) {
    this->normalize_gaussians = normalize_gaussians;
}

void MBTR::set_normalization(string normalization) {
    set<string> options({"l2", "n_atoms", "none", "valle_oganov"});
    if (options.find(normalization) == options.end()) {
        std::ostringstream oss;
        oss << "Unknown normalization option. Please use one of the following: ";
        bool first = true;
        for (auto &it : options) {
            if (!first) {
                oss << ", ";
            }
            oss << "'" << it << "'";
            first = false;
        }
        oss << ".";
        throw invalid_argument(oss.str());
    }
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

double MBTR::get_cutoff() {
    double cutoff = numeric_limits<double>::infinity();
    bool use_perimeter = true;
    if (weighting.contains("function")) {
        string function = weighting["function"].cast<string>();
        if (function == "exp" || function == "exponential") {
            if (weighting.contains("r_cut")) {
                cutoff = weighting["r_cut"].cast<double>();
            } else if (weighting.contains("scale")) {
                double scale = weighting["scale"].cast<double>();
                if (!weighting.contains("threshold")) {
                    throw invalid_argument("Missing value for 'threshold' in the weighting.");
                }
                double threshold = weighting["threshold"].cast<double>();
                cutoff = -log(threshold) / scale;
            }
        } else if (function == "inverse_square") {
            if (weighting.contains("r_cut")) {
                cutoff = weighting["r_cut"].cast<double>();
            }
        } else if (function == "smooth_cutoff") {
            use_perimeter = false;
            if (weighting.contains("r_cut")) {
                cutoff = weighting["r_cut"].cast<double>();
            }
        }
    }

    // In k3, if the distance is defined as the perimeter we half it to get the
    // actual cutoff.
    if (this->k == 3 && use_perimeter) {
        cutoff *= 0.5;
    }

    // For k1, the cutoff is set to zero.
    if (this->k == 1) {
        cutoff = 0;
    }
    return cutoff;
}

void MBTR::validate() {
    this->assert_valle();
    this->assert_weighting();
    this->assert_periodic_weighting();
}

void MBTR::assert_valle() {
    if (this->normalization == "valle_oganov" && !this->periodic) {
        throw invalid_argument("Valle-Oganov normalization does not support non-periodic systems.");
    }
}

void MBTR::assert_periodic_weighting() {
    if (this->periodic && this->k != 1) {
        bool valid = false;
        if (weighting.contains("function")) {
            string function = weighting["function"].cast<string>();
            if (function != "unity") {
                valid = true;
            }
        }
        if (!valid) {
            throw invalid_argument("Periodic systems need to have a weighting function.");
        }
    }
}

void MBTR::assert_weighting() {
    set<string> valid_functions;
    if (this->k == 1) {
        valid_functions = set<string>({"unity"});
    } else if (this->k == 2) {
        valid_functions = set<string>({"unity", "exp", "exponential", "inverse_square"});
    } else {
        valid_functions = set<string>({"unity", "exp", "exponential", "smooth_cutoff"});
    }
    ostringstream os;
    string function = weighting["function"].cast<string>();
    if (valid_functions.find(function) == valid_functions.end()) {
        std::ostringstream oss;
        oss << "Unknown weighting function specified for k=" << this->k << ". Please use one of the following: ";
        bool first = true;
        for (auto &it : valid_functions) {
            if (!first) {
                oss << ", ";
            }
            oss << "'" << it << "'";
            first = false;
        }
        oss << ".";
        throw invalid_argument(oss.str());
    } else {
        if (function == "exp" || function == "exponential") {
            if (!weighting.contains("threshold")) {
                throw invalid_argument("Missing value for 'threshold' in the weighting.");
            }
            if (!weighting.contains("scale") && !weighting.contains("r_cut")) {
                throw invalid_argument("Provide either 'scale' or 'r_cut' in the weighting.");
            }
            if (weighting.contains("scale") && weighting.contains("r_cut")) {
                throw invalid_argument("Provide only 'scale' or 'r_cut' in the weighting, not both.");
            }
        } else if (function == "inverse_square") {
            if (!weighting.contains("r_cut")) {
                throw invalid_argument("Missing value for 'r_cut' in the weighting.");
            }
        } else if (function == "smooth_cutoff") {
            if (!weighting.contains("r_cut")) {
                throw invalid_argument("Missing value for 'r_cut' in the weighting.");
            }
        }
    }
}

py::array_t<int> MBTR::get_species() {return this->species;};
bool MBTR::get_periodic() {return this->periodic;};
map<int, int> MBTR::get_species_index_map() {return this->species_index_map;};
py::dict MBTR::get_geometry() {return geometry;};
py::dict MBTR::get_grid() {return grid;};
py::dict MBTR::get_weighting() {return weighting;};
int MBTR::get_k() {return k;};
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
    // Create mutable and unchecked versions
    py::array_t<int> atomic_numbers = system.atomic_numbers;
    auto out_mu = out.mutable_unchecked<1>();
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();

    // Get grid setup
    double sigma = this->grid["sigma"].cast<double>();
    double min = this->grid["min"].cast<double>();
    double max = this->grid["max"].cast<double>();
    int n = this->grid["n"].cast<int>();
    double dx = (max - min) / (n - 1);
    double start = min - dx/2;

    // Determine the geometry function to use
    string geom_func_name = this->geometry["function"].cast<string>();
    function<double(int)> geom_func;
    if (geom_func_name == "atomic_number") {
        geom_func = geom_atomic_number;
    } else {
        throw invalid_argument("Invalid geometry function for k=1.");
    }

    // Determine the weighting function to use
    string weight_func_name = this->weighting["function"].cast<string>();
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
    // Create mutable and unchecked versions
    auto out_mu = out.mutable_unchecked<1>();
    auto atomic_numbers = system.atomic_numbers;
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();

    // Get grid setup
    double sigma = this->grid["sigma"].cast<double>();
    double min = this->grid["min"].cast<double>();
    double max = this->grid["max"].cast<double>();
    int n = this->grid["n"].cast<int>();
    double dx = (max - min) / (n - 1);
    double start = min - dx/2;

    // Determine the geometry function to use
    string geom_func_name = this->geometry["function"].cast<string>();
    function<double(double)> geom_func;
    if (geom_func_name == "distance") {
        geom_func = geom_distance;
    } else if (geom_func_name == "inverse_distance") {
        geom_func = geom_inverse_distance;
    } else {
        throw invalid_argument("Invalid geometry function for k=2.");
    }

    // Determine the weighting function to use
    string weight_func_name = this->weighting["function"].cast<string>();
    function<double(double)> weight_func;
    if (weight_func_name == "unity") {
        weight_func = weight_unity_k2;
    } else if (weight_func_name == "exp" || weight_func_name == "exponential") {
        double scale = get_scale();
        weight_func = bind(weight_exponential_k2, std::placeholders::_1, scale);
    } else if (weight_func_name == "inverse_square") {
        weight_func = weight_square_k2;
    } else {
        throw invalid_argument("Invalid weighting function for k=2.");
    }

    double cutoff_k2 = this->cutoff;
    auto cell_indices_u = system.cell_indices.unchecked<1>();
    auto system_indices_u = system.indices.unchecked<1>();
    int n_atoms = atomic_numbers.size();
    int n_atoms_original = system.interactive_atoms.size();

    // Loop through all the atoms in the extended cell: TODO: It might be
    // possible to only loop through atoms in the original cell, but then it is
    // hard to figure out which pairs to drop out due to multiplicity caused by
    // periodicity, especially when dealing with cells where the same atom
    // interacts with itself.
    for (int i = 0; i < n_atoms; ++i) {
        CellListResult neighbours_i = cell_list.getNeighboursForIndex(i);
        int n_neighbours = neighbours_i.indices.size();
        // Loop through all neighbours of i
        for (int it = 0; it < n_neighbours; ++it) {
            int j = neighbours_i.indices[it];

            // Distance is symmetric, only consider one way
            if (j <= i) {
                continue;
            }

            // Only consider pairs that have one atom in the original cell
            if (i >= n_atoms_original && j >= n_atoms_original) {
                continue;
            }

            // Early return if distance is bigger than cutoff
            double distance = neighbours_i.distances[it];
            if (distance > cutoff_k2) {
                continue;
            }

            double geom = geom_func(distance);
            double weight = weight_func(distance);

            // When the pair of atoms are in different copies of the cell, the
            // weight is halved. This is done in order to avoid double counting
            // the same distance in the opposite direction. This correction
            // makes periodic cells with different translations equal and also
            // supercells equal to the primitive cell within a constant that is
            // given by the number of repetitions of the primitive cell in the
            // supercell.
            int i_cell = cell_indices_u(i);
            int j_cell = cell_indices_u(j);
            if (i_cell != j_cell) {
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

void MBTR::calculate_k3(py::array_t<double> &out, System &system, CellList &cell_list) {
    // Create mutable and unchecked versions
    auto out_mu = out.mutable_unchecked<1>();
    auto atomic_numbers = system.atomic_numbers;
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();
    auto positions = system.positions;
    auto positions_u = positions.unchecked<2>();

    // Get grid setup
    double sigma = this->grid["sigma"].cast<double>();
    double min =this->grid["min"].cast<double>();
    double max = this->grid["max"].cast<double>();
    int n = this->grid["n"].cast<int>();
    double dx = (max - min) / (n - 1);
    double start = min - dx/2;

    // Determine the geometry function to use
    string geom_func_name = this->geometry["function"].cast<string>();
    function<double(double, double, double)> geom_func;
    if (geom_func_name == "angle") {
        geom_func = geom_angle;
    } else if (geom_func_name == "cosine") {
        geom_func = geom_cosine;
    } else {
        throw invalid_argument("Invalid geometry function for k=3.");
    }

    // Determine the weighting function to use
    bool use_perimeter = true;
    string weight_func_name = this->weighting["function"].cast<string>();
    function<double(double, double, double)> weight_func;
    if (weight_func_name == "unity") {
        weight_func = weight_unity_k3;
    } else if (weight_func_name == "exp" || weight_func_name == "exponential") {
        double scale = get_scale();
        weight_func = bind(
            weight_exponential_k3,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            scale
        );
    } else if (weight_func_name == "smooth_cutoff") {
        use_perimeter = false;
        double sharpness = this->weighting["sharpness"].cast<double>();
        double r_cut = this->weighting["r_cut"].cast<double>();
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

    double cutoff_pairwise = this->cutoff;
    double cutoff_perimeter = this->cutoff * 2;
    auto pos_u = system.positions.unchecked<2>();
    auto cell_indices_u = system.cell_indices.unchecked<1>();
    auto system_indices_u = system.indices.unchecked<1>();
    int n_atoms = atomic_numbers.size();
    int n_atoms_original = system.interactive_atoms.size();

    // Loop through all the atoms in the extended cell. Atom j is the central
    // one. TODO: It might be possible to only loop through atoms in the
    // original cell, but then it is hard to figure out which triplets to drop
    // out due to multiplicity caused by periodicity, especially when dealing
    // with cells where the same atom interacts with itself.
    for (int j = 0; j < n_atoms; ++j) {
        CellListResult neighbours_j = cell_list.getNeighboursForIndex(j);
        int n_neighbours_j = neighbours_j.indices.size();
        // Loop through all neighbours of j
        for (int it_i = 0; it_i < n_neighbours_j; ++it_i) {
            int i = neighbours_j.indices[it_i];
            // Loop through all other neighbours of j
            for (int it_k = 0; it_k < n_neighbours_j; ++it_k) {
                int k = neighbours_j.indices[it_k];
                // Only consider triplets that have at least one atom in the original cell
                if (i >= n_atoms_original && j >= n_atoms_original && k >= n_atoms_original)  {
                    continue;
                }

                if (j == i || k == j || k == i) {
                    continue;
                }

                // The angles are symmetric: angle between ij and jk is the same
                // as the angle between jk and ij. The value is calculated only
                // for the one where where k > i.
                if (k <= i) {
                    continue;
                }

                // Early return if distance is bigger than cutoff
                double distance_ij = neighbours_j.distances[it_i];
                double distance_jk = neighbours_j.distances[it_k];
                if (use_perimeter) {
                    if (distance_ij + distance_jk > cutoff_perimeter) {
                        continue;
                    }
                } else if (distance_ij > cutoff_pairwise || distance_jk > cutoff_pairwise) {
                    continue;
                }

                // The i-k distance is here calculated to check for early
                // return. TODO: One could alternatively check if k is part of
                // i's neighbours: if not, then this triplet can be skipped.
                // This would be possible if e.g. celllist result indices would
                // be an ordered set.
                double d_x = positions_u(i, 0) - positions_u(k, 0);
                double d_y = positions_u(i, 1) - positions_u(k, 1);
                double d_z = positions_u(i, 2) - positions_u(k, 2);
                double distance_ki = sqrt(d_x*d_x + d_y*d_y + d_z*d_z);
                if (use_perimeter && (distance_ij + distance_jk + distance_ki > cutoff_perimeter)) {
                    continue;
                }

                // Calculate geometry value.
                double geom = geom_func(distance_ij, distance_jk, distance_ki);

                // Calculate weight value.
                double weight = weight_func(distance_ij, distance_jk, distance_ki);

                // The contributions are weighted by their multiplicity arising
                // from the translational symmetry. Each triple of atoms is
                // repeated N times in the extended system through translational
                // symmetry. The weight for the angles is thus divided by N so
                // that the multiplication from symmetry is countered. This
                // makes the final spectrum invariant to the selected supercell
                // size and shape after normalization. The number of repetitions
                // N is given by how many unique cell indices (the index of the
                // repeated cell with respect to the original cell at index [0,
                // 0, 0]) are present for the atoms in the triple.
                int i_cell = cell_indices_u(i);
                int j_cell = cell_indices_u(j);
                int k_cell = cell_indices_u(k);
                bool ij_diff = i_cell != j_cell;
                bool jk_diff = j_cell != k_cell;
                bool ki_diff = i_cell != k_cell;
                int diff_sum = (int)ij_diff + (int)jk_diff + (int)ki_diff;
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