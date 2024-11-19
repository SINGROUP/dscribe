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

#include <set>
#include <unordered_map>
#include <cmath>
#include "descriptorglobal.h"
#include "geometry.h"

using namespace std;

DescriptorGlobal::DescriptorGlobal(bool periodic, string average, double cutoff, string normalization)
    : Descriptor(periodic, average, cutoff, normalization)
{
}

void DescriptorGlobal::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    bool return_descriptor,
    bool return_derivatives
)
{
    // Extend system if periodicity is requested.
    auto pbc_u = pbc.unchecked<1>();
    bool is_periodic = this->periodic && (pbc_u(0) || pbc_u(1) || pbc_u(2));
    System system = is_periodic
        ? extend_system(positions, atomic_numbers, cell, pbc, this->cutoff)
        : System(positions, atomic_numbers, cell);
    this->create(out, system, return_descriptor, return_derivatives);
}

void DescriptorGlobal::create(
    py::array_t<double> out, 
    System &system,
    bool return_descriptor,
    bool return_derivatives
)
{
    // Calculate neighbours with a cell list
    CellList cell_list(system.positions, this->cutoff);
    this->create(out, system, cell_list, return_descriptor, return_derivatives);
}

void DescriptorGlobal::derivatives_numerical(
    py::array_t<double> derivatives, 
    py::array_t<double> descriptor, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    py::array_t<int> indices,
    bool return_descriptor
)
{
    int n_atoms = atomic_numbers.size();
    int n_features = this->get_number_of_features();
    auto derivatives_mu = derivatives.mutable_unchecked<3>();
    auto indices_u = indices.unchecked<1>();
    auto pbc_u = pbc.unchecked<1>();

    // Extend the system if it is periodic
    bool is_periodic = this->periodic && (pbc_u(0) || pbc_u(1) || pbc_u(2));
    System system = is_periodic
      ? extend_system(positions, atomic_numbers, cell, pbc, this->cutoff)
      : System(positions, atomic_numbers, cell);
    int n_copies = system.atomic_numbers.size() / atomic_numbers.size();
    auto positions_mu = system.positions.mutable_unchecked<2>();

    // Pre-calculate cell list for atoms
    CellList cell_list_atoms(system.positions, this->cutoff);

    // Calculate the desciptor value if requested
    if (return_descriptor) {
        this->create(descriptor, system, cell_list_atoms, true, false);
    }

    // Central finite difference with error O(h^2)
    double h = 0.0001;
    vector<double> coefficients = {-1.0/2.0, 1.0/2.0};
    vector<double> displacement = {-1.0, 1.0};

    // Loop over all atoms
    for (int i_pos=0; i_pos < indices_u.size(); ++i_pos) {
        int i_atom = indices_u(i_pos);

        // Create a list of all atom indices that should be moved. For periodic
        // systems all of the periodic copies need to be moved as well.
        vector<int> i_atom_indices(n_copies);
        for (int i = 0; i < n_copies; ++i) {
            i_atom_indices[i] = i_atom + i*n_atoms;
        }

        // Create a copy of the original atom position(s). These will be used
        // to reset the positions after each displacement.
        py::array_t<double> pos({n_copies, 3});
        auto pos_mu = pos.mutable_unchecked<2>();
        for (size_t i_copy = 0; i_copy < i_atom_indices.size(); ++i_copy) {
            int j_copy = i_atom_indices[i_copy];
            for (int i = 0; i < 3; ++i) {
                pos_mu(i_copy, i) = positions_mu(j_copy, i);
            }
        }

        for (int i_comp=0; i_comp < 3; ++i_comp) {
            for (int i_stencil=0; i_stencil < 2; ++i_stencil) {

                // Introduce the displacement(s). Displacement are done for all
                // periodic copies as well.
                for (size_t i_copy = 0; i_copy < i_atom_indices.size(); ++i_copy) {
                    int j_copy = i_atom_indices[i_copy];
                    positions_mu(j_copy, i_comp) = pos_mu(i_copy, i_comp) + h*displacement[i_stencil];
                }

                // Initialize temporary numpy array for storing the descriptor
                // for this stencil point
                double* dTemp = new double[n_features]();
                py::array_t<double> d({n_features}, dTemp);
                auto d_mu = d.mutable_unchecked<1>();

                // Calculate descriptor value
                this->create(d, system, cell_list_atoms, true, false);

                // Add value to final derivative array
                double coeff = coefficients[i_stencil];
                for (int i_feature=0; i_feature < n_features; ++i_feature) {
                    double value = coeff*d_mu(i_feature);
                    derivatives_mu(i_pos, i_comp, i_feature) = derivatives_mu(i_pos, i_comp, i_feature) + value;
                }

                delete [] dTemp;
            }

            for (int i_feature=0; i_feature < n_features; ++i_feature) {
                derivatives_mu(i_pos, i_comp, i_feature) = derivatives_mu(i_pos, i_comp, i_feature) / h;
            }

            // Return position(s) back to original value for next component.
            for (size_t i_copy = 0; i_copy < i_atom_indices.size(); ++i_copy) {
                int j_copy = i_atom_indices[i_copy];
                positions_mu(j_copy, i_comp) = pos_mu(i_copy, i_comp);
            }
        }
    }
}
