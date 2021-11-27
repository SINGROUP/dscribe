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
#include "descriptor.h"
#include "geometry.h"

using namespace std;

Descriptor::Descriptor(bool periodic, string average, double cutoff)
    : periodic(periodic)
    , average(average)
    , cutoff(cutoff)
{
}

/**
 * The general idea: each atom for which a derivative is requested is
 * "wiggled" with central finite difference. The following tricks are used
 * to speed up the calculation:
 *
 *  - The CellList for positions is calculated only once and passed to the
 *    create-method.
 *  - Only centers within the cutoff distance from the wiggled atom are
 *    taken into account by calculating a separate CellList for the
 *    centers. Note that this optimization cannot be naively applied when
 *    averaging centers.
 *  - Atoms for which there are no neighouring centers are skipped.
 *
 *  TODO:
 *  - Symmetry of the derivatives should be taken into account (derivatives
 *    of [i, j] is -[j, i] AND species position should be swapped)
 *  - Using only the local centers in the averaged case.
 *
 *  Notice that these optimization are NOT valid:
 *  - Self-derivatives are NOT always zero, only zero for l=0. The shape of
 *    the atomic density changes as atoms move around.
 */
void Descriptor::derivatives_numerical(
    py::array_t<double> derivatives, 
    py::array_t<double> descriptor, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    py::array_t<double> centers,
    py::array_t<int> center_indices,
    py::array_t<int> indices,
    bool attach,
    bool return_descriptor
) const
{
    int n_copies = 1;
    int n_atoms = atomic_numbers.size();
    int n_features = this->get_number_of_features();
    auto derivatives_mu = derivatives.mutable_unchecked<4>();
    auto indices_u = indices.unchecked<1>();
    auto pbc_u = pbc.unchecked<1>();
    py::array_t<int> center_true_indices;
    py::array_t<double> centers_extended;

    // Extend the system if it is periodic
    bool is_periodic = this->periodic && (pbc_u(0) || pbc_u(1) || pbc_u(2));
    if (is_periodic) {
        ExtendedSystem system_extension = extend_system(positions, atomic_numbers, cell, pbc, this->cutoff);
        n_copies = system_extension.atomic_numbers.size()/atomic_numbers.size();
        positions = system_extension.positions;
        atomic_numbers = system_extension.atomic_numbers;
    }
    auto positions_mu = positions.mutable_unchecked<2>();

    // Pre-calculate cell list for atoms
    CellList cell_list_atoms(positions, this->cutoff);

    // Calculate the desciptor value if requested. This needs to be done with
    // the non-extended centers, but extended system.
    if (return_descriptor) {
        this->create(descriptor, positions, atomic_numbers, centers, cell_list_atoms);
    }

    // Create the extended centers for periodic systems.
    if (is_periodic) {
        ExtendedSystem center_extension = extend_system(centers, center_indices, cell, pbc, this->cutoff);
        centers_extended = center_extension.positions;
        center_true_indices = center_extension.indices;
    } else {
        centers_extended = centers;
        center_true_indices = center_indices;
    }

    auto centers_u = centers.unchecked<2>();

    // Pre-calculate cell list for centers.
    CellList cell_list_centers(centers_extended, this->cutoff);

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

        // Get all centers within the cutoff range from the current atom.
        double ix = positions_mu(i_atom, 0);
        double iy = positions_mu(i_atom, 1);
        double iz = positions_mu(i_atom, 2);
        vector<int> centers_local_idx = cell_list_centers.getNeighboursForPosition(ix, iy, iz).indices;

        // The periodically repeated centers are already correctly found from
        // the extended center list. They just need to be mapped into the
        // correct centers in the original list, as the original center
        // position should be used (the atoms are periodically repeated and all
        // periodic copies are moved for the finite difference).
        if (is_periodic) {
            auto center_true_indices_u = center_true_indices.unchecked<1>();
            set<int> centers_set;
            for (size_t i=0; i < centers_local_idx.size(); ++i) {
                int ext_index = centers_local_idx[i];
                int true_index = center_true_indices_u(ext_index);
                centers_set.insert(true_index);
            }
            centers_local_idx = vector<int>(centers_set.begin(), centers_set.end()); 
        }

        // If there are no centers within the cutoff radius from the atom, the
        // calculation is skipped.
        int n_locals = centers_local_idx.size();
        if (n_locals == 0) {
            continue;
        }

        // If attach = true, find the center(s) that need to be moved together
        // with this atom.
        vector<int> centers_to_move;
        if (attach) {
            auto center_true_indices_u = center_true_indices.unchecked<1>();
            for (int i_local = 0; i_local < n_locals; ++i_local) {
                int i_local_idx = centers_local_idx[i_local];
                int true_index = center_true_indices_u(i_local_idx);
                if (true_index == i_atom) {
                    centers_to_move.push_back(i_local);
                }
            }
        }

        // If averaging is not performed, create a list of the local center
        // coordinates. Only this subset needs to be used. For averaged
        // calculation it is not as simple, so for now we simply use all
        // centers.
        int n_centers;
        py::array_t<double> centers_local_pos;
        if (this->average == "off") {
            centers_local_pos = py::array_t<double>({n_locals, 3});
            auto centers_local_pos_mu = centers_local_pos.mutable_unchecked<2>();
            for (int i_local = 0; i_local < n_locals; ++i_local) {
                int i_local_idx = centers_local_idx[i_local];
                for (int i_comp = 0; i_comp < 3; ++i_comp) {
                    centers_local_pos_mu(i_local, i_comp) = centers_u(i_local_idx, i_comp);
                }
            }
            n_centers = n_locals;
        } else {
            centers_local_pos = centers;
            centers_local_idx = vector<int>{0};
            n_centers = 1;
        }
        auto centers_local_pos_mu = centers_local_pos.mutable_unchecked<2>();

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

        // If attach = true, create a copy of the original center position(s).
        // These will be used to reset the center positions after each
        // displacement.
        py::array_t<double> centers_moved({(int)centers_to_move.size(), 3});
        auto centers_moved_mu = centers_moved.mutable_unchecked<2>();
        if (attach) {
            for (size_t i_copy = 0; i_copy < centers_to_move.size(); ++i_copy) {
                int j_copy = centers_to_move[i_copy];
                for (int i = 0; i < 3; ++i) {
                    centers_moved_mu(i_copy, i) = centers_local_pos_mu(j_copy, i);
                }
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

                // If attach = true, we also move the center(s) that are
                // attached to this atom.
                if (attach) {
                    for (size_t i_copy = 0; i_copy < centers_to_move.size(); ++i_copy) {
                        int j_copy = centers_to_move[i_copy];
                        centers_local_pos_mu(j_copy, i_comp) = centers_moved_mu(i_copy, i_comp) + h*displacement[i_stencil];
                    }
                }

                // Initialize temporary numpy array for storing the descriptor
                // for this stencil point
                double* dTemp = new double[n_centers*n_features]();
                py::array_t<double> d({n_centers, n_features}, dTemp);

                // Calculate descriptor value
                this->create(d, positions, atomic_numbers, centers_local_pos, cell_list_atoms);
                auto d_u = d.unchecked<2>();

                // Add value to final derivative array
                double coeff = coefficients[i_stencil];
                for (int i_local=0; i_local < n_centers; ++i_local) {
                    int i_center = centers_local_idx[i_local];
                    for (int i_feature=0; i_feature < n_features; ++i_feature) {
                        double value = coeff*d_u(i_local, i_feature);
                        derivatives_mu(i_center, i_pos, i_comp, i_feature) = derivatives_mu(i_center, i_pos, i_comp, i_feature) + value;
                    }
                }

                delete [] dTemp;
            }

            for (int i_local=0; i_local < n_centers; ++i_local) {
                int i_center = centers_local_idx[i_local];
                for (int i_feature=0; i_feature < n_features; ++i_feature) {
                    derivatives_mu(i_center, i_pos, i_comp, i_feature) = derivatives_mu(i_center, i_pos, i_comp, i_feature) / h;
                }
            }

            // Return position(s) back to original value for next component.
            for (size_t i_copy = 0; i_copy < i_atom_indices.size(); ++i_copy) {
                int j_copy = i_atom_indices[i_copy];
                positions_mu(j_copy, i_comp) = pos_mu(i_copy, i_comp);
            }

            // If attach = true, return center(s) back to original value for
            // next component.
            if (attach) {
                for (size_t i_copy = 0; i_copy < centers_to_move.size(); ++i_copy) {
                    int j_copy = centers_to_move[i_copy];
                    centers_local_pos_mu(j_copy, i_comp) = centers_moved_mu(i_copy, i_comp);
                }
            }
        }
    }
}
