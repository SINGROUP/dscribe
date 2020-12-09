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
#include "descriptor.h"

using namespace std;

Descriptor::Descriptor(string average, double cutoff)
    : average(average)
    , cutoff(cutoff)
{
}

void Descriptor::derivatives_numerical(
    py::array_t<double> out_d, 
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> centers,
    py::array_t<int> indices,
    bool return_descriptor
) const
{
    // Calculate neighbours with a cell list
    CellList cell_list(positions, this->cutoff);

    int n_features = this->get_number_of_features();
    auto out_d_mu = out_d.mutable_unchecked<4>();
    auto indices_u = indices.unchecked<1>();
    auto positions_mu = positions.mutable_unchecked<2>();
    int n_centers;
    if (this->average != "off") {
        n_centers = 1;
    } else {
        n_centers = centers.shape(0);
    }

    // Calculate the desciptor value if requested
    if (return_descriptor) {
        this->create(out, positions, atomic_numbers, centers, cell_list);
    }
    
    // Central finite difference with error O(h^2)
    double h = 0.0001;
    vector<double> coefficients = {-1.0/2.0, 1.0/2.0};
    vector<double> displacement = {-1.0, 1.0};

    for (int i_pos=0; i_pos < indices_u.size(); ++i_pos) {

        // Create a copy of the original atom position
        py::array_t<double> pos(3);
        auto pos_mu = pos.mutable_unchecked<1>();
        for (int i = 0; i < 3; ++i) {
            pos_mu(i) = positions_mu(i_pos, i);
        }

        for (int i_comp=0; i_comp < 3; ++i_comp) {
            for (int i_stencil=0; i_stencil < 2; ++i_stencil) {

                // Introduce the displacement
                positions_mu(i_pos, i_comp) = pos_mu(i_comp) + h*displacement[i_stencil];

                // Initialize temporary numpy array for storing the descriptor
                // for this stencil point
                double* dTemp = new double[n_centers*n_features]();
                py::array_t<double> d({n_centers, n_features}, dTemp);

                // Calculate descriptor value
                this->create(d, positions, atomic_numbers, centers, cell_list);
                auto d_u = d.unchecked<2>();

                // Add value to final derivative array
                double coeff = coefficients[i_stencil];
                for (int i_center=0; i_center < n_centers; ++i_center) {
                    for (int i_feature=0; i_feature < n_features; ++i_feature) {
                        out_d_mu(i_center, i_pos, i_comp, i_feature) = (out_d_mu(i_center, i_pos, i_comp, i_feature) + coeff*d_u(i_center, i_feature));
                    }
                }
                delete [] dTemp;
            }
            for (int i_center=0; i_center < n_centers; ++i_center) {
                for (int i_feature=0; i_feature < n_features; ++i_feature) {
                    out_d_mu(i_center, i_pos, i_comp, i_feature) = out_d_mu(i_center, i_pos, i_comp, i_feature) / h;
                }
            }

            // Return position back to original value for next component
            positions_mu(i_pos, i_comp) = pos_mu(i_comp);
        }
    }
}
