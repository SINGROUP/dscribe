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
#include <math.h>

using namespace std;

CoulombMatrix::CoulombMatrix(
    unsigned int n_atoms_max,
    string permutation,
    double sigma,
    int seed
)
    : DescriptorGlobal(true)
    , n_atoms_max(n_atoms_max)
    , permutation(permutation)
    , sigma(sigma)
    , seed(seed)
{
}

void CoulombMatrix::create_raw(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    CellList &cell_list
) const
{
    // Calculate all pairwise distances. CellList is the generic container that
    // can calculate distances even if no cutoff is available.
    py::array_t<double> matrix = cell_list.getAllDistances();
    auto matrix_mu = matrix.mutable_unchecked<2>();
    auto out_mu = out.mutable_unchecked<1>();
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();

    // Construct matrix
    int n_atoms = atomic_numbers.shape(0);
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
        this->getEigenspectrum(matrix, out);
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
    py::array_t<double> matrix,
    py::array_t<double> out
) const
{
}

void CoulombMatrix::sort(
    py::array_t<double> matrix
) const
{
}

void CoulombMatrix::sortRandomly(
    py::array_t<double> matrix
) const
{
}

int CoulombMatrix::get_number_of_features() const
{
    return this->permutation == "eigenspectrum"
        ? this->n_atoms_max
        : this->n_atoms_max * this->n_atoms_max;
}
