#include "celllist.h"
#include <algorithm>
#include <utility>
#include <map>
#include <utility>
#include <iostream>
#include <math.h>

using namespace std;

CellList::CellList(py::array_t<double> positions, double cutoff)
    : positions(positions.unchecked<2>())
    , cutoff(cutoff)
    , cutoffSquared(cutoff*cutoff)
{
    this->init();
}

void CellList::init() {
    // Find cell limits
    this->xmin = this->xmax = this->positions(0, 0);
    this->ymin = this->ymax = this->positions(0, 1);
    this->zmin = this->zmax = this->positions(0, 2);
    for (ssize_t i = 0; i < this->positions.shape(0); i++) {
        double x = this->positions(i, 0);
        double y = this->positions(i, 1);
        double z = this->positions(i, 2);
        if (x < this->xmin) {
            this->xmin = x;
        };
        if (x > this->xmax) {
            this->xmax = x;
        };
        if (y < this->ymin) {
            this->ymin = y;
        };
        if (y > this->ymax) {
            this->ymax = y;
        };
        if (z < this->zmin) {
            this->zmin = z;
        };
        if (z > this->zmax) {
            this->zmax = z;
        };
    };
    double padding = 0.001;
    this->xmin -= padding;
    this->xmax += padding;
    this->ymin -= padding;
    this->ymax += padding;
    this->zmin -= padding;
    this->zmax += padding;

    // Determine amount and size of bins
    this->nx = ceil((this->xmax - this->xmin)/this->cutoff);
    this->ny = ceil((this->ymax - this->ymin)/this->cutoff);
    this->nz = ceil((this->zmax - this->zmin)/this->cutoff);
    this->dx = (this->xmax - this->xmin)/this->nx;
    this->dy = (this->ymax - this->ymin)/this->ny;
    this->dz = (this->zmax - this->zmin)/this->nz;

    // Initialize the bin data structure. It is a 4D vector.
    this->bins = vector<vector<vector<vector<int>>>>(this->nx, vector<vector<vector<int>>>(this->ny, vector<vector<int>>(this->nz, vector<int>())));

    // Fill the bins with atom indices
    for (ssize_t index = 0; index < this->positions.shape(0); index++) {
        double x = this->positions(index, 0);
        double y = this->positions(index, 1);
        double z = this->positions(index, 2);

        // Get bin index
        int i = (x - this->xmin)/this->dx;
        int j = (y - this->ymin)/this->dy;
        int k = (z - this->zmin)/this->dz;

        // Add atom index to the bin
        this->bins[i][j][k].push_back(index);
    };
}

pair<vector<int>, vector<double>> CellList::getNeighboursForPosition(const double x, const double y, const double z) const
{
    // The indices of the neighbouring atoms
    vector<int> neighbours;
    vector<double> distances;

    // Find bin for the given position
    int i0 = (x - this->xmin)/this->dx;
    int j0 = (y - this->ymin)/this->dy;
    int k0 = (z - this->zmin)/this->dz;

    // Find neighbouring bins, check whether current bin is on boundary
    int istart = i0 > 0 ? i0-1 : 0;
    int iend = i0 < this->nx-1 ? i0+2 : this->nx;
    int jstart = j0 > 0 ? j0-1 : 0;
    int jend = j0 < this->ny-1 ? j0+2 : this->ny;
    int kstart = k0 > 0 ? k0-1 : 0;
    int kend = k0 < this->nz-1 ? k0+2 : this->nz;

    // Loop over neighbouring bins
    for (int i = istart; i < iend; i++){
        for (int j = jstart; j < jend; j++){
            for (int k = kstart; k < kend; k++){

                // For each atom in the current bin, calculate the actual distance
                vector<int> binIndices = this->bins[i][j][k];
                for (auto &idx : binIndices) {
                    double ix = this->positions(idx, 0);
                    double iy = this->positions(idx, 1);
                    double iz = this->positions(idx, 2);
                    double deltax = x - ix;
                    double deltay = y - iy;
                    double deltaz = z - iz;
                    double distanceSquared = deltax*deltax + deltay*deltay + deltaz*deltaz;
                    if (distanceSquared <= this->cutoffSquared) {
                        neighbours.push_back(idx);
                        distances.push_back(sqrt(distanceSquared));
                    }
                }
            }
        }
    }
    return make_pair(neighbours, distances);
}

pair<vector<int>, vector<double>> CellList::getNeighboursForIndex(const int idx) const
{
    double x = this->positions(idx, 0);
    double y = this->positions(idx, 1);
    double z = this->positions(idx, 2);
    pair<vector<int>, vector<double>> result = this->getNeighboursForPosition(x, y, z);

    // Remove self from neighbours
    for (int i=0; i < result.first.size(); ++i) {
        if (result.first[i] == idx) {
            result.first.erase(result.first.begin() + i);
            result.second.erase(result.second.begin() + i);
            break;
        }
    }
    return result;
}
