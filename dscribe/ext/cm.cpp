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
#include "cm.h"
#include "celllist.h"
#include "geometry.h"
#include <math.h>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

CoulombMatrix::CoulombMatrix(
    unsigned int n_atoms_max,
    string permutation,
    double sigma,
    int seed
)
    : DescriptorGlobal(false)
    , n_atoms_max(n_atoms_max)
    , permutation(permutation)
    , sigma(sigma)
    , seed(seed)
{
}

void CoulombMatrix::create_raw(
    py::detail::unchecked_mutable_reference<double, 1> &out_mu, 
    py::detail::unchecked_reference<double, 2> &positions_u,
    py::detail::unchecked_reference<int, 1> &atomic_numbers_u,
    CellList &cell_list
) const
{
    // Calculate all pairwise distances.
    py::array_t<double> matrix = distances(positions_u);
    auto matrix_mu = matrix.mutable_unchecked<2>();

    // Construct matrix
    int n_atoms = atomic_numbers_u.shape(0);
    for (int i = 0; i < n_atoms; ++i) {
        for (int j = i; j < n_atoms; ++j) {
            if (j == i) {
                matrix_mu(i, j) = 0.5 * pow(atomic_numbers_u(i), 2.4);
            } else {
                double value = atomic_numbers_u(i) * atomic_numbers_u(j) / matrix_mu(i, j);
                matrix_mu(i, j) = value;
                matrix_mu(j, i) = value;
            }
        }
    }

    // Handle the permutation option
    if (this->permutation == "eigenspectrum") {
        this->getEigenspectrum(matrix_mu, out_mu, n_atoms);
    } else {
        if (this->permutation == "sorted") {
            this->sort(matrix);
        } else if (this->permutation == "random") {
            this->sortRandomly(matrix);
        }
        // Flatten
        int k = 0;
        for (int i = 0; i < n_atoms; ++i) {
            for (int j = 0; j < n_atoms; ++j) {
                out_mu(k) = matrix_mu(i, j);
                ++k;
            }
        }
    }
}

void CoulombMatrix::getEigenspectrum(
    py::detail::unchecked_mutable_reference<double, 2> &matrix_mu,
    py::detail::unchecked_mutable_reference<double, 1> &out_mu,
    int n_atoms
) const
{
    // Calculate eigenvalues with Eigen
    MatrixXd A(n_atoms, n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        for (int j = i; j < n_atoms; ++j) {
            // Only the lower triangular part is referenced
            A(j, i) = matrix_mu(i, j);
        }
    }
    SelfAdjointEigenSolver<MatrixXd> eigensolver(A, EigenvaluesOnly);
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();

    // Sort the values in descending order by absolute value
    std::sort(
        eigenvalues.data(),
        eigenvalues.data()+eigenvalues.size(),
        [ ]( const double& lhs, const double& rhs ) {
            return abs(lhs) > abs(rhs);
        }
    );

    // Copy to output
    for (int i = 0; i < n_atoms; ++i) {
        out_mu[i] = eigenvalues(i);
    }
}

void CoulombMatrix::sort(
    py::array_t<double> &matrix
) const
{
}

void CoulombMatrix::sortRandomly(
    py::array_t<double> &matrix
) const
{
}

int CoulombMatrix::get_number_of_features() const
{
    return this->permutation == "eigenspectrum"
        ? this->n_atoms_max
        : this->n_atoms_max * this->n_atoms_max;
}
