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
#include <unordered_map>
#include <unordered_set>
#include "descriptor.h"

using namespace std;

Descriptor::Descriptor(string average, double cutoff)
    : average(average)
    , cutoff(cutoff)
{
}

//void Descriptor::derivatives_numerical(
    //py::array_t<double> out_d, 
    //py::array_t<double> out, 
    //py::array_t<double> positions,
    //py::array_t<int> atomic_numbers,
    //py::array_t<double> centers,
    //py::array_t<int> indices,
    //bool return_descriptor
//) const
//{
    //// Calculate neighbours with a cell list
    //CellList cell_list(positions, this->cutoff);

    //int n_features = this->get_number_of_features();
    //auto out_d_mu = out_d.mutable_unchecked<4>();
    //auto indices_u = indices.unchecked<1>();
    //auto positions_mu = positions.mutable_unchecked<2>();
    //int n_centers;
    //if (this->average != "off") {
        //n_centers = 1;
    //} else {
        //n_centers = centers.shape(0);
    //}

    //// Calculate the desciptor value if requested
    //if (return_descriptor) {
        //this->create(out, positions, atomic_numbers, centers, cell_list);
    //}
    
    //// Central finite difference with error O(h^2)
    //double h = 0.0001;
    //vector<double> coefficients = {-1.0/2.0, 1.0/2.0};
    //vector<double> displacement = {-1.0, 1.0};

    //for (int i_pos=0; i_pos < indices_u.size(); ++i_pos) {

        //// Create a copy of the original atom position
        //py::array_t<double> pos(3);
        //auto pos_mu = pos.mutable_unchecked<1>();
        //for (int i = 0; i < 3; ++i) {
            //pos_mu(i) = positions_mu(i_pos, i);
        //}

        //for (int i_comp=0; i_comp < 3; ++i_comp) {
            //for (int i_stencil=0; i_stencil < 2; ++i_stencil) {

                //// Introduce the displacement
                //positions_mu(i_pos, i_comp) = pos_mu(i_comp) + h*displacement[i_stencil];

                //// Initialize temporary numpy array for storing the descriptor
                //// for this stencil point
                //double* dTemp = new double[n_centers*n_features]();
                //py::array_t<double> d({n_centers, n_features}, dTemp);

                //// Calculate descriptor value
                //this->create(d, positions, atomic_numbers, centers, cell_list);
                //auto d_u = d.unchecked<2>();

                //// Add value to final derivative array
                //double coeff = coefficients[i_stencil];
                //for (int i_center=0; i_center < n_centers; ++i_center) {
                    //for (int i_feature=0; i_feature < n_features; ++i_feature) {
                        //out_d_mu(i_center, i_pos, i_comp, i_feature) = (out_d_mu(i_center, i_pos, i_comp, i_feature) + coeff*d_u(i_center, i_feature));
                    //}
                //}
                //delete [] dTemp;
            //}
            //for (int i_center=0; i_center < n_centers; ++i_center) {
                //for (int i_feature=0; i_feature < n_features; ++i_feature) {
                    //out_d_mu(i_center, i_pos, i_comp, i_feature) = out_d_mu(i_center, i_pos, i_comp, i_feature) / h;
                //}
            //}

            //// Return position back to original value for next component
            //positions_mu(i_pos, i_comp) = pos_mu(i_comp);
        //}
    //}
//}

void Descriptor::derivatives_numerical(
    py::array_t<double> out_d, 
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> center_pos,
    py::array_t<int> center_indices,
    py::array_t<int> indices,
    bool return_descriptor
) const
{
    int n_features = this->get_number_of_features();
    auto out_d_mu = out_d.mutable_unchecked<4>();
    auto center_pos_u = center_pos.unchecked<2>();
    auto center_indices_u = center_indices.unchecked<1>();
    auto indices_u = indices.unchecked<1>();
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();
    auto positions_u = positions.unchecked<2>();

    // Calculate the descriptor value if requested
    if (return_descriptor) {
        this->create(out, positions, atomic_numbers, center_pos);
    }

    // Create mappings between center index and atom index and vice versa. The
    // order of centers and indices can be arbitrary, and not all centers
    // correspond to atoms.
    unordered_map<int, int> index_atom_map;
    unordered_map<int, int> index_center_map;
    unordered_map<int, int> center_atom_map;
    unordered_map<int, int> atom_center_map;
    for (int i=0; i < center_indices.size(); ++i) {
        int index = center_indices_u(i);
        if (index != -1) {
            index_center_map[index] = i;
        }
    }
    for (int i=0; i < indices.size(); ++i) {
        int index = indices_u(i);
        index_atom_map[index] = i;
    }
    for (int i=0; i < center_indices.size(); ++i) {
        int index = center_indices_u(i);
        if (index != -1 && index_atom_map.find(index) != index_atom_map.end()) {
            center_atom_map[i] = index_atom_map[index];
        }
    }
    for (int i=0; i < indices.size(); ++i) {
        int index = indices_u(i);
        if (index_center_map.find(index) != index_center_map.end()) {
            atom_center_map[i] = index_center_map[index];
        }
    }
    cout << "" << endl;
    for(auto it = center_atom_map.begin(); it != center_atom_map.end(); ++it){
        cout << it->first << " " << it->second << " " << endl;
    }

    // Central finite difference with error O(h^2)
    double h = 0.0001;
    vector<double> coefficients = {-1.0/2.0, 1.0/2.0};
    vector<double> displacement = {-1.0, 1.0};

    int n_indices = indices.shape(0);
    CellList cell_list_centers(center_pos, this->cutoff);
    for (int i_in=0; i_in < n_indices; ++i_in) {
        // Get the neighbouring centers/atoms for the "wiggled" atom i. The
        // neighbours are gathered into a list and they act as SOAP centers for
        // which the derivative with respect to the movement of atom i is
        // calculated.
        int i_out = indices_u(i_in);
        double ix = positions_u(i_out, 0);
        double iy = positions_u(i_out, 1);
        double iz = positions_u(i_out, 2);
        CellListResult result = cell_list_centers.getNeighboursForPosition(ix, iy, iz);
        vector<int> neighbours = result.indices;

        // Construct a possibly smaller neighbour list by taking symmetry and
        // self-forces into account. Only possible if the center corresponds to
        // an atom.
        vector<int> locals;
        unordered_set<int> symmetric;
        cout << "Moving atom: " << i_out << endl;
        bool not_center = atom_center_map.find(i_in) == atom_center_map.end();
        for (int i_neighbour=0; i_neighbour < neighbours.size(); ++i_neighbour) {
            int ii_neighbour = neighbours[i_neighbour];
            int iii_neighbour = center_indices_u(ii_neighbour);
            cout << "Neighbour center: " << ii_neighbour << endl;
            cout << "Neighbour index: " << iii_neighbour << endl;

            // If the wiggled atom does does not correspond to any center, or
            // if neighbouring center does not correspond to any atom,
            // calculate this pair.
            if (not_center || center_atom_map.find(ii_neighbour) == center_atom_map.end()) {
                locals.push_back(ii_neighbour);
            // Otherwise utilize symmetry and ignore self-interaction.
            } else if (iii_neighbour > i_out) {
                locals.push_back(ii_neighbour);
                symmetric.insert(ii_neighbour);
            }
        }

        // When there are no local atoms that should be calculated, skip.
        int n_locals = locals.size();
        if (n_locals == 0) {
            continue;
        }
        for (int i_local = 0; i_local < n_locals; ++i_local) {
            cout << "Neighbour atom: " << locals[i_local] << endl;
        }

        // Create temporary array of local positions
        py::array_t<double> centers_local({n_locals, 3});
        auto centers_local_mu = centers_local.mutable_unchecked<2>();
        for (int i_local = 0; i_local < n_locals; ++i_local) {
            int ii_local = locals[i_local];
            for (int i_comp = 0; i_comp < 3; ++i_comp) {
                centers_local_mu(i_local, i_comp) = center_pos_u(ii_local, i_comp);
            }
        }

        // Create temporary array for the wiggled position and atomic number
        py::array_t<double> position_local({1, 3});
        auto position_local_mu = position_local.mutable_unchecked<2>();
        py::array_t<double> atomic_number_local(1);
        auto atomic_number_local_mu = atomic_number_local.mutable_unchecked<1>();
        atomic_number_local_mu(0) = atomic_numbers_u(i_out);

        for (int i_comp=0; i_comp < 3; ++i_comp) {
            for (int i_stencil=0; i_stencil < 2; ++i_stencil) {

                // Introduce the displacement
                position_local_mu(0, i_comp) = positions_u(i_out, i_comp) + h*displacement[i_stencil];

                // Initialize temporary numpy array for storing the
                // descriptor for this stencil point. This array needs to
                // be initialized with zeros, thus the slightly different
                // syntax
                double* dTemp = new double[n_locals*n_features]();
                py::array_t<double> d({n_locals, n_features}, dTemp);

                // Calculate descriptor value
                this->create(d, position_local, atomic_number_local, centers_local);
                auto d_u = d.unchecked<2>();

                // Add value to final derivative array
                double coeff = coefficients[i_stencil];
                for (int i_local = 0; i_local < n_locals; ++i_local) {
                    int j_in = locals[i_local];
                    for (int i_feature=0; i_feature < n_features; ++i_feature) {
                        out_d_mu(j_in, i_in, i_comp, i_feature) = (out_d_mu(j_in, i_in, i_comp, i_feature) + coeff*d_u(i_local, i_feature));
                        if (symmetric.find(j_in) != symmetric.end()) {
                            out_d_mu(atom_center_map[i_in], center_atom_map[j_in], i_comp, i_feature) = -out_d_mu(j_in, i_in, i_comp, i_feature);
                        }
                    }
                }
                delete [] dTemp;
            }
            for (int i_local = 0; i_local < n_locals; ++i_local) {
                int j_in = locals[i_local];
                for (int i_feature=0; i_feature < n_features; ++i_feature) {
                    out_d_mu(j_in, i_in, i_comp, i_feature) = out_d_mu(j_in, i_in, i_comp, i_feature) / h;
                    if (symmetric.find(j_in) != symmetric.end()) {
                        out_d_mu(atom_center_map[i_in], center_atom_map[j_in], i_comp, i_feature) = out_d_mu(atom_center_map[i_in], center_atom_map[j_in], i_comp, i_feature) / h;
                    }
                }
            }

            // Return position back to original value for next component
            position_local_mu(0, i_comp) = positions_u(i_out, i_comp);
        }
    }
}
