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

#include "geometry.h"

namespace py = pybind11;
using namespace std;

inline vector<double> cross(const vector<double>& a, const vector<double>& b) {
    return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}

inline double dot(const vector<double>& a, const vector<double>& b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline double norm(const vector<double>& a) {
    double accum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        accum += a[i] * a[i];
    }
    return sqrt(accum);
};

ExtendedSystem extend_system(
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    double cutoff)
{
    if (cutoff < 0) {
        throw invalid_argument("Cutoff must be positive.");
    }
    // Determine the upper limit of how many copies we need in each cell vector
    // direction. We take as many copies as needed to reach the radial cutoff.
    // Notice that we need to use vectors that are perpendicular to the cell
    // vectors to ensure that the correct atoms are included for non-cubic
    // cells.
    auto positions_u = positions.unchecked<2>();
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();
    auto cell_u = cell.unchecked<2>();
    auto pbc_u = pbc.unchecked<1>();
    vector<double> a = {cell_u(0, 0), cell_u(0, 1), cell_u(0, 2)};
    vector<double> b = {cell_u(1, 0), cell_u(1, 1), cell_u(1, 2)};
    vector<double> c = {cell_u(2, 0), cell_u(2, 1), cell_u(2, 2)};

    vector<double> p1 = cross(b, c);
    vector<double> p2 = cross(c, a);
    vector<double> p3 = cross(a, b);

    // Projections of basis vectors onto perpendicular vectors.
    double p1_coeff = dot(a, p1) / dot(p1, p1);
    double p2_coeff = dot(b, p2) / dot(p2, p2);
    double p3_coeff = dot(c, p3) / dot(p3, p3);
    for(double &x : p1) { x *= p1_coeff; }
    for(double &x : p2) { x *= p2_coeff; }
    for(double &x : p3) { x *= p3_coeff; }
    vector<vector<double>> vectors = {p1, p2, p3};

    // Figure out how many copies to take per basis vector. Determined by how
    // many perpendicular projections fit into the cutoff distance.
    vector<int> n_copies_axis(3);
    vector<vector<int>> multipliers;
    for (int i=0; i < 3; ++i) {
        if (pbc_u(i)) {
            double length = norm(vectors[i]);
            double factor = cutoff/length;
            int multiplier = (int)ceil(factor);
            n_copies_axis[i] = multiplier;

            // Store multipliers explicitly into a list in an order that keeps the
            // original system in the same place both in space and in index.
            vector<int> multiples;
            for (int j=0; j < multiplier + 1; ++j) {
                multiples.push_back(j);
            }
            for (int j=-multiplier; j < 0; ++j) {
                multiples.push_back(j);
            }
            multipliers.push_back(multiples);
        } else {
            n_copies_axis[i] = 0;
            multipliers.push_back(vector<int>{0});
        }
    }

    // Calculate the extended system positions.
    int n_rep = (2*n_copies_axis[0]+1)*(2*n_copies_axis[1]+1)*(2*n_copies_axis[2]+1);
    int n_atoms = atomic_numbers.size();
    py::array_t<double> ext_pos({n_atoms*n_rep, 3});
    py::array_t<int> ext_atomic_numbers({n_atoms*n_rep});
    py::array_t<int> ext_indices({n_atoms*n_rep});
    auto ext_pos_mu = ext_pos.mutable_unchecked<2>();
    auto ext_atomic_numbers_mu = ext_atomic_numbers.mutable_unchecked<1>();
    auto ext_indices_mu = ext_indices.mutable_unchecked<1>();
    int i_copy = 0;
    int a_limit = multipliers[0].size();
    int b_limit = multipliers[1].size();
    int c_limit = multipliers[2].size();
    for (int i=0; i < a_limit; ++i) {
        int a_multiplier = multipliers[0][i];
        for (int j=0; j < b_limit; ++j) {
            int b_multiplier = multipliers[1][j];
            for (int k=0; k < c_limit; ++k) {
                int c_multiplier = multipliers[2][k];

                // Precalculate the added vector. It will be used many times.
                vector<double> addition(3, 0);
                for (int m=0; m < 3; ++m) {
                    addition[m] += a_multiplier*a[m] + b_multiplier*b[m] + c_multiplier*c[m];
                };

                // Store the positions, atomic numbers and indices
                for (int l=0; l < n_atoms; ++l) {
                    int index = i_copy*n_atoms + l;
                    ext_atomic_numbers_mu(index) = atomic_numbers_u(l);
                    ext_indices_mu(index) = l;
                    for (int m=0; m < 3; ++m) {
                        ext_pos_mu(index, m) = positions_u(l, m) + addition[m];
                    }
                }
                ++i_copy;
            }
        }
    }

    return ExtendedSystem{ext_pos, ext_atomic_numbers, ext_indices};
}

py::array_t<double> distancesNumpy(py::detail::unchecked_reference<double, 2> &positions_u)
{
    int n_atoms = positions_u.shape(0);
    py::array_t<double> distances({n_atoms, n_atoms});
    auto distances_mu = distances.mutable_unchecked<2>();
    for (int i = 0; i < n_atoms; ++i) {
        for (int j = i; j < n_atoms; ++j) {
            double dx = positions_u(i, 0) - positions_u(j, 0);
            double dy = positions_u(i, 1) - positions_u(j, 1);
            double dz = positions_u(i, 2) - positions_u(j, 2);
            double distance = sqrt(dx*dx + dy*dy + dz*dz);
            distances_mu(i, j) = distance;
            distances_mu(j, i) = distance;
        }
    }
    return distances;
}

MatrixXd distancesEigen(py::detail::unchecked_reference<double, 2> &positions_u)
{
    int n_atoms = positions_u.shape(0);
    MatrixXd distances(n_atoms, n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        for (int j = i; j < n_atoms; ++j) {
            double dx = positions_u(i, 0) - positions_u(j, 0);
            double dy = positions_u(i, 1) - positions_u(j, 1);
            double dz = positions_u(i, 2) - positions_u(j, 2);
            double distance = sqrt(dx*dx + dy*dy + dz*dz);
            distances(i, j) = distance;
            distances(j, i) = distance;
        }
    }
    return distances;
}
