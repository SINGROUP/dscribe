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
            const py::dict k1,
            const py::dict k2,
            const py::dict k3,
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
        void set_k1(py::dict k1);
        void set_k2(py::dict k2);
        void set_k3(py::dict k3);
        void set_normalize_gaussians(bool normalize_gaussians);
        void set_normalization(string normalization);

        /**
         * Getters
         */
        py::array_t<int> get_species();
        bool get_periodic();
        py::dict get_k1();
        py::dict get_k2();
        py::dict get_k3();
        bool get_normalize_gaussians();
        string get_normalization();
        map<int, int> get_species_index_map();

        py::dict k1;
        py::dict k2;
        py::dict k3;
        bool normalize_gaussians;
        string normalization;
        py::array_t<int> species;
        bool periodic;

    private:
        double cutoff_k2;
        double cutoff_k3;
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
        void assert_periodic_weighting(py::dict &k);
        /**
         * Checks that the weighting is defined correctly in the given
         * configuration dict.
         */
        void assert_weighting(py::dict &k);
        /**
         * Gets the radial cutoff value from the given configuration dict.
         */
        double get_cutoff(py::dict &k);
        inline vector<double> gaussian(double center, double weight, double start, double dx, double sigma, int n);
        void calculate_k1(py::array_t<double> &out, System &system);
        void calculate_k2(py::array_t<double> &out, System &system, CellList &cell_list);
        void calculate_k3(py::array_t<double> &out, System &system, CellList &cell_list);
        int get_number_of_k1_features() const;
        int get_number_of_k2_features() const;
        int get_number_of_k3_features() const;
        void normalize_output(py::array_t<double> &out);
};

#endif
