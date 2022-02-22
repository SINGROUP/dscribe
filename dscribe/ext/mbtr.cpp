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
#include "mbtr.h"
#include "constants.h"

using namespace std;

inline double weight_unity_k1(const int &atomic_number)
{
    return 1;
}

inline double weight_unity_k2(const int &i, const int &j, const vector<vector<double> > &distances)
{
    return 1;
}

inline double weight_unity_k3(const int &i, const int &j, const int &k, const vector<vector<double> > &distances)
{
    return 1;
}

inline double weight_exponential_k2(const int &i, const int &j, const vector<vector<double> > &distances, double scale)
{
    double dist = distances[i][j];
    double expValue = exp(-scale*dist);
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

inline double weight_square_k2(const int &i, const int &j, const vector<vector<double> > &distances)
{
    double dist = distances[i][j];
    double value = 1/(dist*dist);
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

inline double geom_distance(const int &i, const int &j, const vector<vector<double> > &distances)
{
    double dist = distances[i][j];
    return dist;
}

inline double geom_inverse_distance(const int &i, const int &j, const vector<vector<double> > &distances)
{
    double dist = geom_distance(i, j, distances);
    double invDist = 1/dist;
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
    , normalization(normalization)
{
    this->set_species(species);
    this->set_k1(k1);
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
void MBTR::set_normalize_gaussians(bool normalize_gaussians) {this->normalize_gaussians = normalize_gaussians;};
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
    double normalization = this->normalize_gaussians
      ? 1.0/2.0
      : 1.0/2.0 * sigmasqrt2 * SQRT2;
    for (auto &it : cdf) {
        it = weight * normalization * (1.0 + erf((x-center)/sigmasqrt2));
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

    // Get k1 grid setup
    double sigma = this->k1["grid"]["sigma"].cast<double>();
    double min = this->k1["grid"]["min"].cast<double>();
    double max = this->k1["grid"]["max"].cast<double>();
    int n = this->k1["grid"]["n"].cast<int>();
    double dx = (max - min) / (n - 1);
    double sigmasqrt2 = sigma * sqrt(2.0);
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
        vector<double> gauss = gaussian(geom, weight, start, dx, sigmasqrt2, n);

        // Get the index of the present elements in the final vector
        int i_index = this->species_index_map.at(i_z);

        // Sum gaussian into output
        for (int j=0; j < gauss.size(); ++j) {
            out_mu[i_index * n + j] += gauss[j];
        }
    }
}
