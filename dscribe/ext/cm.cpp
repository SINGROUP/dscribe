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
#include <numeric>
#include <iostream>

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
    , generator(seed)
{
}

void CoulombMatrix::create_raw(
    py::detail::unchecked_mutable_reference<double, 1> &out_mu, 
    py::detail::unchecked_reference<double, 2> &positions_u, 
    py::detail::unchecked_reference<int, 1> &atomic_numbers_u,
    CellList &cell_list
)
{
    // Calculate all pairwise distances.
    int n_atoms = atomic_numbers_u.shape(0);
    MatrixXd matrix = distancesEigen(positions_u);

    // Construct matrix
    for (int i = 0; i < n_atoms; ++i) {
        for (int j = i; j < n_atoms; ++j) {
            if (j == i) {
                matrix(i, j) = 0.5 * pow(atomic_numbers_u(i), 2.4);
            } else {
                double value = atomic_numbers_u(i) * atomic_numbers_u(j) / matrix(i, j);
                matrix(i, j) = value;
                matrix(j, i) = value;
            }
        }
    }

    // Handle the permutation option
    if (this->permutation == "eigenspectrum") {
        this->get_eigenspectrum(matrix, out_mu);
    } else {
        if (this->permutation == "sorted_l2") {
            this->sort(matrix, false);
        } else if (this->permutation == "random") {
            this->sort(matrix, true);
        }
        // Flatten. Notice that we have to flatten in a way that takes into
        // account the size of the entire matrix.
        int k = 0;
        int diff = this->n_atoms_max - n_atoms;
        for (int i = 0; i < n_atoms; ++i) {
            for (int j = 0; j < n_atoms; ++j) {
                out_mu(k) = matrix(i, j);
                ++k;
            }
            k += diff;
        }
    }
}

void CoulombMatrix::get_eigenspectrum(
    const Ref<const MatrixXd> &matrix,
    py::detail::unchecked_mutable_reference<double, 1> &out_mu
)
{
    // Calculate eigenvalues with Eigen
    SelfAdjointEigenSolver<MatrixXd> eigensolver(matrix, EigenvaluesOnly);
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
    for (int i = 0; i < matrix.cols(); ++i) {
        out_mu[i] = eigenvalues(i);
    }
}

void CoulombMatrix::sort(Ref<MatrixXd> matrix, bool noise)
{
    // Calculate row norms
    VectorXd norms = matrix.rowwise().norm();
    int n_atoms = matrix.rows();

    // Introduce noise with norm as mean and sigma as standard deviation.
    if (noise) {
        for (int i = 0; i < norms.size(); ++i) {
            normal_distribution<double> distribution(norms(i), this->sigma);
            norms(i) = distribution(this->generator);
        }
    }

    // Sort the pairs by norm.
    VectorXi indices(n_atoms);
    iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&norms](int a, int b) { return norms(a) > norms(b); } );

    // Sort matrix in place. Notice that we sort both rows and columns. This way
    // the interpretation of the matrix as pairwise interaction of atoms is
    // still valid. NOTE: Getting eigen to correctly sort both columns and rows
    // with the same order was non-trivial, but the following syntax seems to
    // work fine.
    PermutationMatrix<Dynamic> P(indices);
    matrix = P.transpose() * matrix * P;
}

int CoulombMatrix::get_number_of_features() const
{
    return this->permutation == "eigenspectrum"
        ? this->n_atoms_max
        : this->n_atoms_max * this->n_atoms_max;
}
