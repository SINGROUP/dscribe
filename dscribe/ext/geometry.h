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

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>
#include <stdexcept>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

namespace py = pybind11;
using namespace Eigen;
using namespace std;

struct ExtendedSystem {
    py::array_t<double> positions;
    py::array_t<int> atomic_numbers;
    py::array_t<int> indices;
};

inline vector<double> cross(const vector<double>& a, const vector<double>& b);
inline double dot(const vector<double>& a, const vector<double>& b);
inline double norm(const vector<double>& a);

/**
 * Used to periodically extend an atomic system in order to take into account
 * periodic copies beyond the given unit cell.
 * 
 * @param positions Cartesian positions of the original system.
 * @param atomic_numbers Atomic numbers of the original system.
 * @param cell Unit cell of the original system.
 * @param pbc Periodic boundary conditions (array of three booleans) of the original system.
 * @param cutoff Radial cutoff value for determining extension size.
 *
 * @return Instance of ExtendedSystem.
 */
ExtendedSystem extend_system(
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    double cutoff);

/**
 * Used to calculate the full distance matrix (numpy) for the given positions.
 *
 * @param positions Cartesian positions in a <n_atoms, 3> array.
 *
 * @return Pairwise distances in an <n_atoms, n_atoms> array.
 */
py::array_t<double> distancesNumpy(py::detail::unchecked_reference<double, 2> &positions_u);

/**
 * Used to calculate the full distance matrix (eigen) for the given positions.
 *
 * @param positions Cartesian positions in a <n_atoms, 3> array.
 *
 * @return Pairwise distances in an <n_atoms, n_atoms> array.
 */
MatrixXd distancesEigen(py::detail::unchecked_reference<double, 2> &positions_u);

#endif
