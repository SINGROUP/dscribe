#ifndef CELLLIST_H
#define CELLLIST_H

#include <pybind11/numpy.h>
#include "binning.h"
#include <vector>

namespace py = pybind11;
using namespace std;

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
        pair<vector<int>,vector<double>> getNeighboursForPosition(const double x, const double y, const double z) const;
        /**
         * Get the indices of atoms within the radial cutoff distance from the
         * given atomic index.
         *
         * @param i Index of the atom for which neighbours are queried for.
         */
        pair<vector<int>,vector<double>> getNeighboursForIndex(const int i) const;

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
