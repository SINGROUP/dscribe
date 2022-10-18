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
#ifndef MBTR_H
#define MBTR_H

#include <string>
#include <map>
#include <pybind11/numpy.h>
#include "descriptorglobal.h"

namespace py = pybind11;
using namespace std;

/**
 * Many-body Tensor Representation (MBTR) descriptor.
 */
class MBTR: public DescriptorGlobal {
    public:
        /**
         * Constructor, see the python docs for more details about variables.
         */
        MBTR(
            const py::dict geometry,
            const py::dict grid,
            const py::dict weighting,
            bool normalize_gaussians,
            string normalization,
            py::array_t<int> species,
            bool periodic
        );

        /**
         * For creating feature vectors.
         */
        void create(py::array_t<double> &out, System &system, CellList &cell_list);

        /**
         * Get the number of features.
         */
        int get_number_of_features() const;

        /**
         * Get location of k1 features.
         */
        pair<int, int> get_location(int z1);

        /**
         * Get location of k2 features.
         */
        pair<int, int> get_location(int z1, int z2);

        /**
         * Get location of k3 features.
         */
        pair<int, int> get_location(int z1, int z2, int z3);

        /**
         * Setters
         */
        void set_species(py::array_t<int> species);
        void set_periodic(bool periodic);
        void set_geometry(py::dict geometry);
        void set_grid(py::dict grid);
        void set_weighting(py::dict weighting);
        void set_normalize_gaussians(bool normalize_gaussians);
        void set_normalization(string normalization);

        /**
         * Getters
         */
        py::array_t<int> get_species();
        bool get_periodic();
        py::dict get_geometry();
        py::dict get_grid();
        py::dict get_weighting();
        int get_k();
        bool get_normalize_gaussians();
        string get_normalization();
        map<int, int> get_species_index_map();

        py::dict geometry;
        py::dict grid;
        py::dict weighting;
        bool normalize_gaussians;
        string normalization;
        py::array_t<int> species;
        bool periodic;

    private:
        int k;
        map<int, int> species_index_map;
        map<int, int> index_species_map;
        /**
         * Performs all assertions.
         */
        void validate();
        /**
         * Checks that the configuration is valid for valle-oganoff.
         */
        void assert_valle();
        /**
         * Checks that the weighting is defined correctly when periodicity is
         * taken into account.
         */
        void assert_periodic_weighting();
        /**
         * Checks that the weighting is defined correctly.
         */
        void assert_weighting();
        /**
         * Gets the radial cutoff value.
         */
        double get_cutoff();
        /**
         * Gets the exponential scaling factor.
         */
        double get_scale();
        inline void add_gaussian(
            double center,
            double weight,
            double start,
            double dx,
            double sigma,
            int n,
            const pair<int, int> &loc,
            py::detail::unchecked_mutable_reference<double, 1> &out
        );
        void calculate_k1(py::array_t<double> &out, System &system);
        void calculate_k2(py::array_t<double> &out, System &system, CellList &cell_list);
        void calculate_k3(py::array_t<double> &out, System &system, CellList &cell_list);
        int get_number_of_k1_features() const;
        int get_number_of_k2_features() const;
        int get_number_of_k3_features() const;
        void normalize_output(py::array_t<double> &out, int start = 0, int end = -1);
};

#endif
