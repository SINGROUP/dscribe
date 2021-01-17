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

#include <iostream>
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
 *  - Using only the local centers is more difficult in the averaged case,
 *    especially in the inner-averaging mode. Thus these optimization are
 *    simply left out.
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
    bool return_descriptor
) const
{
    int n_copies = 1;
    int n_atoms = atomic_numbers.size();
    int n_features = this->get_number_of_features();
    int n_c = center_indices.size();
    auto derivatives_mu = derivatives.mutable_unchecked<4>();
    auto indices_u = indices.unchecked<1>();
    auto center_indices_u = center_indices.unchecked<1>();
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
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();

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
            for (int i=0; i < centers_local_idx.size(); ++i) {
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

        // Create a copy of the original atom position(s). These will be used
        // to reset the positions after each displacement.
        py::array_t<double> pos({n_copies, 3});
        auto pos_mu = pos.mutable_unchecked<2>();
        for (int i_copy = 0; i_copy < i_atom_indices.size(); ++i_copy) {
            int j_copy = i_atom_indices[i_copy];
            for (int i = 0; i < 3; ++i) {
                pos_mu(i_copy, i) = positions_mu(j_copy, i);
            }
        }

        for (int i_comp=0; i_comp < 3; ++i_comp) {
            for (int i_stencil=0; i_stencil < 2; ++i_stencil) {

                // Introduce the displacement(s). Displacement are done for all
                // periodic copies as well.
                for (int i_copy = 0; i_copy < i_atom_indices.size(); ++i_copy) {
                    int j_copy = i_atom_indices[i_copy];
                    positions_mu(j_copy, i_comp) = pos_mu(i_copy, i_comp) + h*displacement[i_stencil];
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
            for (int i_copy = 0; i_copy < i_atom_indices.size(); ++i_copy) {
                int j_copy = i_atom_indices[i_copy];
                positions_mu(j_copy, i_comp) = pos_mu(i_copy, i_comp);
            }
        }
    }
}

//void Descriptor::derivatives_numerical(
    //py::array_t<double> out_d, 
    //py::array_t<double> out, 
    //py::array_t<double> positions,
    //py::array_t<int> atomic_numbers,
    //py::array_t<double> cell,
    //py::array_t<bool> pbc,
    //py::array_t<double> centers,
    //py::array_t<int> center_indices,
    //py::array_t<int> indices,
    //bool return_descriptor
//) const
//{
    //// The general idea: each atom for which a derivative is requested is
    //// "wiggled" with central finite difference. The following tricks are used
    //// to speed up the calculation:
    ////  - The CellList for positions is calculated only once and passed to the
    ////    create-method.
    ////  - Only centers within the cutoff distance from the wiggled atom are
    ////    taken into account by calculating a separate CellList for the
    ////    centers. Note that this optimization cannot be naively applied when
    ////    averaging centers.
    ////  - Atoms for which there are no neighouring centers are skipped.
    ////  TODO:
    ////  - Symmetry of the derivatives is taken into account (derivatives of
    ////    [i, j] is -[j, i] AND species position should be swapped)
    ////  - Using symmetry and removing self-interaction in the averaged case is
    ////    much more difficult, especially in the inner-averaging mode. Thus these
    ////    optimization are simply left out.
    ////
    ////  Notice that these optimization are NOT valid:
    ////  - Self-derivatives are NOT always zero, only zero for l=0. The shape of
    ////    the atomic density changes as atoms move around.
    //int n_features = this->get_number_of_features();
    //int n_c = center_indices.size();
    //auto out_d_mu = out_d.mutable_unchecked<4>();
    //auto indices_u = indices.unchecked<1>();
    //auto center_indices_u = center_indices.unchecked<1>();
    //auto pbc_u = pbc.unchecked<1>();
    //py::array_t<int> center_true_indices;
    //int n_copies = 1;
    //int n_atoms = atomic_numbers.size();

    //// Create the extended system for periodic systems. 
    //bool is_periodic = this->periodic && (pbc_u(0) || pbc_u(1) || pbc_u(2));
    //if (is_periodic) {
        //ExtendedSystem system_extension = extend_system(positions, atomic_numbers, cell, pbc, this->cutoff);
        //positions = system_extension.positions;
        //n_copies = system_extension.atomic_numbers.size()/atomic_numbers.size();
        //atomic_numbers = system_extension.atomic_numbers;
    //}
    //auto positions_mu = positions.mutable_unchecked<2>();
    //auto atomic_numbers_u = atomic_numbers.unchecked<1>();
    ////cout << "Number of copies: " << n_copies << endl;
    ////py::detail::unchecked_reference<int, 1> center_true_indices_u = is_periodic ? center_true_indices.unchecked<1>(): center_indices.unchecked<1>();

    //// Pre-calculate cell list for atoms
    //CellList cell_list_atoms(positions, this->cutoff);

    //// Calculate the desciptor value if requested. This needs to be done with
    //// the non-extended centers, but extended system.
    //if (return_descriptor) {
        //this->create(out, positions, atomic_numbers, centers, cell_list_atoms);
    //}

    //// Create the extended centers for periodic systems.
    //if (is_periodic) {
        //ExtendedSystem center_extension = extend_system(centers, center_indices, cell, pbc, this->cutoff);
        //centers = center_extension.positions;
        //center_true_indices = center_extension.indices;
    //}
    //auto centers_u = centers.unchecked<2>();

    //// Pre-calculate cell list for centers.
    //CellList cell_list_centers(centers, this->cutoff);

    //// TODO: These are needed for tracking symmetrical values.
    //// Create mappings between center index and atom index and vice versa. The
    //// order of centers and indices can be arbitrary, and not all centers
    //// correspond to atoms.
    //unordered_map<int, int> index_atom_map;
    //unordered_map<int, int> center_atom_map;
    ////unordered_map<int, int> index_center_map;
    ////unordered_map<int, int> atom_center_map;
    ////for (int i=0; i < center_indices.size(); ++i) {
        ////int index = center_indices_u(i);
        ////if (index != -1) {
            ////index_center_map[index] = i;
        ////}
    ////}
    ////for (int i=0; i < indices.size(); ++i) {
        ////int index = indices_u(i);
        ////if (index_center_map.find(index) != index_center_map.end()) {
            ////atom_center_map[i] = index_center_map[index];
        ////}
    ////}
    //for (int i=0; i < indices.size(); ++i) {
        //int index = indices_u(i);
        //index_atom_map[index] = i;
    //}
    //for (int i=0; i < center_indices.size(); ++i) {
        //int index = center_indices_u(i);
        //if (index != -1 && index_atom_map.find(index) != index_atom_map.end()) {
            //center_atom_map[i] = index_atom_map[index];
        //}
    //}

    //// Central finite difference with error O(h^2)
    //double h = 0.0001;
    //vector<double> coefficients = {-1.0/2.0, 1.0/2.0};
    //vector<double> displacement = {-1.0, 1.0};

    //// Loop over all atoms
    //for (int i_idx=0; i_idx < indices_u.size(); ++i_idx) {
        //int i_atom = indices_u(i_idx);
        ////cout << "Atom index: " << i_atom << endl;

        //// Find all atom indices that should be moved. For periodic systems
        //// there may be multiple.
        //vector<int> i_atom_indices(n_copies);
        //for (int i = 0; i < n_copies; ++i) {
            //i_atom_indices[i] = i_atom + i*n_atoms;
        //}

        //// Check whether the atom has any centers within radius. If not, the
        //// calculation is skipped.
        //double ix = positions_mu(i_atom, 0);
        //double iy = positions_mu(i_atom, 1);
        //double iz = positions_mu(i_atom, 2);
        //vector<int> centers_local_idx = cell_list_centers.getNeighboursForPosition(ix, iy, iz).indices;

        ////cout << "Neighbours:" << endl;
        ////for (int i=0; i < centers_local_idx.size(); ++i) {
            ////cout << "  Extended index: " << centers_local_idx[i] << endl;
            ////cout << "  Position: " << centers_u(centers_local_idx[i], 0) << "," << centers_u(centers_local_idx[i], 1) << "," << centers_u(centers_local_idx[i], 2) << endl;
        ////}

        //// The periodically repeated centers are already correctly found from
        //// the extended center list. They just need to be mapped into the
        //// correct centers in the original list, as the original center
        //// position should be used (the atoms are periodically repeated and all
        //// periodic copies are moved for the finite difference).
        //if (is_periodic) {
            //auto center_true_indices_u = center_true_indices.unchecked<1>();
            //set<int> centers_set;
            //for (int i=0; i < centers_local_idx.size(); ++i) {
                //int ext_index = centers_local_idx[i];
                //int true_index = center_true_indices_u(ext_index);
                //centers_set.insert(true_index);
            //}
            //centers_local_idx = vector<int>(centers_set.begin(), centers_set.end()); 
        //}

        //int n_locals = centers_local_idx.size();
        //if (n_locals == 0) {
            //continue;
        //}

        //// When averaging is not performed, only use the local centers.
        //// TODO Remove half of symmetrical pairs
        //int n_centers;
        //py::array_t<double> centers_local_pos;
        //if (this->average == "off") {
            ////bool not_center = atom_center_map.find(i_atom) == atom_center_map.end();
            ////unordered_set<int> symmetric;
            ////vector<int> locals;
            ////for (int i = 0; i < centers_local_idx.size(); ++i) {
                ////int local_idx = centers_local_idx[i];
                ////auto center_atom_idx = center_atom_map.find(local_idx);
                ////if (center_atom_idx == center_atom_map.end()) {
                    ////locals.push_back(local_idx);
                ////} else if (center_atom_idx->second != i_atom) {
                    ////locals.push_back(local_idx);
                ////}
                //////if (not_center || center_atom_idx == center_atom_map.end()) {
                    //////locals.push_back(local_idx);
                //////} else if (center_atom_idx->second > i_atom) {
                    //////locals.push_back(local_idx);
                    //////symmetric.insert(local_idx);
                //////}
            ////}
            ////centers_local_idx = locals;
            //n_locals = centers_local_idx.size();
            //if (n_locals == 0) {
                //continue;
            //}

            //// Create a new list containing only the nearby centers.
            //centers_local_pos = py::array_t<double>({n_locals, 3});
            //auto centers_local_pos_mu = centers_local_pos.mutable_unchecked<2>();
            ////cout << "True center indices: " << endl;
            //for (int i_local = 0; i_local < n_locals; ++i_local) {
                //int i_local_idx = centers_local_idx[i_local];
                ////cout << i_local_idx << endl;
                //for (int i_comp = 0; i_comp < 3; ++i_comp) {
                    //centers_local_pos_mu(i_local, i_comp) = centers_u(i_local_idx, i_comp);
                //}
            //}
            ////cout << "True center positions: " << endl;
            ////for (int i = 0; i < centers_local_pos.request().shape[0]; ++i) {
                ////cout << "  " << centers_local_pos_mu(i, 0) << "," << centers_local_pos_mu(i, 1) << "," << centers_local_pos_mu(i, 2) << endl;
            ////}
            //n_centers = n_locals;
        //} else {
            //centers_local_pos = centers;
            //centers_local_idx = vector<int>{0};
            //n_centers = 1;
        //}

        //// DEBUG: USE ALL CENTERS
        ////py::array_t<double> centers_local_pos = centers;
        ////vector<int> centers_local_idx;
        ////for (int i=0; i < n_c; ++i) {centers_local_idx.push_back(i);}
        ////int n_centers = n_c;

        //// Create a copy of the original atom position(s). These will be used
        //// to reset the positions after each displacement.
        //py::array_t<double> pos({n_copies, 3});
        //auto pos_mu = pos.mutable_unchecked<2>();
        //for (int i_copy = 0; i_copy < i_atom_indices.size(); ++i_copy) {
            //int j_copy = i_atom_indices[i_copy];
            //for (int i = 0; i < 3; ++i) {
                //pos_mu(i_copy, i) = positions_mu(j_copy, i);
            //}
        //}

        ////cout << "Atoms that are displaced: " << endl;
        ////for (int i_copy = 0; i_copy < i_atom_indices.size(); ++i_copy) {
            ////int j_copy = i_atom_indices[i_copy];
            ////cout << "  Indices: " << i_copy << ", " << j_copy << endl;
            ////cout << "  Position: " << pos_mu(i_copy, 0) << ", " << pos_mu(i_copy, 1) << ", " << pos_mu(i_copy, 2) << endl;
            ////cout << "  Atomic number: " << atomic_numbers_u(j_copy) << endl;
        ////}

        //for (int i_comp=0; i_comp < 3; ++i_comp) {
            //for (int i_stencil=0; i_stencil < 2; ++i_stencil) {

                //// Introduce the displacement(s). Displacement are done for all
                //// periodic copies as well.
                //for (int i_copy = 0; i_copy < i_atom_indices.size(); ++i_copy) {
                    //int j_copy = i_atom_indices[i_copy];
                    //positions_mu(j_copy, i_comp) = pos_mu(i_copy, i_comp) + h*displacement[i_stencil];
                //}

                //// Initialize temporary numpy array for storing the descriptor
                //// for this stencil point
                //double* dTemp = new double[n_centers*n_features]();
                //py::array_t<double> d({n_centers, n_features}, dTemp);

                //// Calculate descriptor value
                //this->create(d, positions, atomic_numbers, centers_local_pos, cell_list_atoms);
                //auto d_u = d.unchecked<2>();

                //// Add value to final derivative array
                //double coeff = coefficients[i_stencil];
                //for (int i_local=0; i_local < n_centers; ++i_local) {
                    //int i_center = centers_local_idx[i_local];
                    //for (int i_feature=0; i_feature < n_features; ++i_feature) {
                        //double value = coeff*d_u(i_local, i_feature);
                        //out_d_mu(i_center, i_idx, i_comp, i_feature) = out_d_mu(i_center, i_idx, i_comp, i_feature) + value;
                        ////if (symmetric.find(i_center) != symmetric.end()) {
                            ////out_d_mu(atom_center_map[i_idx], center_atom_map[i_center], i_comp, i_feature) = out_d_mu(atom_center_map[i_idx], center_atom_map[i_center], i_comp, i_feature) - value;
                        ////}
                    //}
                //}

                //delete [] dTemp;
            //}

            //for (int i_local=0; i_local < n_centers; ++i_local) {
                //int i_center = centers_local_idx[i_local];
                //for (int i_feature=0; i_feature < n_features; ++i_feature) {
                    //out_d_mu(i_center, i_idx, i_comp, i_feature) = out_d_mu(i_center, i_idx, i_comp, i_feature) / h;
                    ////if (symmetric.find(i_center) != symmetric.end()) {
                        ////out_d_mu(atom_center_map[i_idx], center_atom_map[i_center], i_comp, i_feature) = out_d_mu(atom_center_map[i_idx], center_atom_map[i_center], i_comp, i_feature) / h;
                    ////}
                //}
            //}

            //// Return position(s) back to original value for next component.
            //for (int i_copy = 0; i_copy < i_atom_indices.size(); ++i_copy) {
                //int j_copy = i_atom_indices[i_copy];
                //positions_mu(j_copy, i_comp) = pos_mu(i_copy, i_comp);
            //}
        //}
    //}
    //// Write the symmetrical values as a post-processing step if they are
    //// present.
//}
