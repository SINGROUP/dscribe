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

#ifndef CELLLIST_H
#define CELLLIST_H

#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;
using namespace std;

struct CellListResult {
    vector<int> indices;
    vector<double> distances;
    vector<double> distancesSquared;
};

/**
 * For calculating pairwise distances using a cell list.
 */
class CellList {
    public:
        /**
         * Constructor
         *
         * @param positions Atomic positions in cartesian coordinates.
         * @param atomicNumbers Atomic numbers.
         */
        CellList(py::array_t<double> positions, double cutoff);
        /**
         * Get the indices of atoms within the radial cutoff distance from the
         * given position.
         *
         * @param x Cartesian x-coordinate.
         * @param y Cartesian y-coordinate.
         * @param z Cartesian z-coordinate.
         */
        CellListResult getNeighboursForPosition(const double x, const double y, const double z) const;
        /**
         * Get the indices of atoms within the radial cutoff distance from the
         * given atomic index. The given index is not included in the returned
         * values.
         *
         * @param i Index of the atom for which neighbours are queried for.
         */
        CellListResult getNeighboursForIndex(const int i) const;

    private:
        /**
         * Used to initialize the cell list. Querying for distances is only
         * possible after this initialization.
         */
        void init();

        const py::detail::unchecked_reference<double, 2> positions;
        const double cutoff;
        const double cutoffSquared;
        double xmin;
        double xmax;
        double ymin;
        double ymax;
        double zmin;
        double zmax;
        double dx;
        double dy;
        double dz;
        int nx;
        int ny;
        int nz;
        vector<vector<vector<vector<int>>>> bins;
};

#endif
