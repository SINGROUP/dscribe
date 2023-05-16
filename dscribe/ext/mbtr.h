#ifndef MBTR_H
#define MBTR_H

#include <vector>
#include <map>
#include <tuple>
#include <math.h>
#include <string>
#include <numeric>
#include <utility>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <numeric>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;
#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

typedef tuple<int> index1d;
typedef tuple<int, int> index2d;
typedef tuple<int, int, int> index3d;

/**
 * Implementation for the performance-critical parts of MBTR.
 */
class MBTR {

    public:
        /**
         * Constructor
         *
         * @param positions Atomic positions in cartesian coordinates.
         * @param atomicNumbers Atomic numbers.
         * @param atomicNumberToIndexMap Mapping between atomic numbers and
         * their position in the final MBTR vector.
         * @param interactionLimit The number of atoms that are interacting.
         * The atoms with indices < interactionLimit are considered to be
         * interacting with other atoms.
         * @param cellIndices 3D index of the periodically repeated cell in
         * which each atom is in the system
         * @param local Whether a local or a global MBTR is calculated.
         */
        MBTR(map<int,int> atomicNumberToIndexMap, int interactionLimit,  vector<vector<int>> cellIndices);
        void getK1(py::array_t<double> &descriptor, const vector<int> &Z, const string &geomFunc, const string &weightFunc, const map<string, double> &parameters, double min, double max, double sigma, int n);
        void getK2(py::array_t<double> &descriptor, py::array_t<double> &derivatives, bool return_descriptor, bool return_derivatives, const vector<int> &Z, const vector<vector<double>> &positions, const vector<vector<double>> &distances, const vector<vector<int>> &neighbours, const string &geomFunc, const string &weightFunc, const map<string, double> &parameters, double min, double max, double sigma, int n);
        void getK3(py::array_t<double> &descriptor, py::array_t<double> &derivatives, bool return_descriptor, bool return_derivatives, const vector<int> &Z, const vector<vector<double>> &positions, const vector<vector<double>> &distances, const vector<vector<int>> &neighbours, const string &geomFunc, const string &weightFunc, const map<string, double> &parameters, double min, double max, double sigma, int n);
        vector<map<string, vector<double>>> getK2Local(const vector<int> &indices, const vector<int> &Z, const vector<vector<double>> &distances, const vector<vector<int>> &neighbours, const string &geomFunc, const string &weightFunc, const map<string, double> &parameters, double min, double max, double sigma, int n);
        vector<map<string, vector<double>>> getK3Local(const vector<int> &indices, const vector<int> &Z, const vector<vector<double>> &distances, const vector<vector<int>> &neighbours, const string &geomFunc, const string &weightFunc, const map<string, double> &parameters, double min, double max, double sigma, int n);
        vector<double> gaussian(double center, double weight, double start, double dx, double sigmasqrt2, int n);
        vector<double> xgaussian(double center, double weight, double start, double dx, double sigma, int n);
        
    private:
        const map<int,int> atomicNumberToIndexMap;
        const int interactionLimit;
        const vector<vector<int> > cellIndices;

        /**
         * Calculates the geometry function based on atomic numbers defined for
         * k=1.
         *
         * @param i Index of first atom.
         *
         * @return Atomic number for the given index.
         */
        double k1GeomAtomicNumber(const int &i, const vector<int> &Z);

        /**
         * Weighting of 1. Usually used for finite small
         * systems.
         *
         * @param i Index of first atom.
         *
         * @return Weight of 1.
         */
        double k1WeightUnity(const int &i);
        /**
         * Calculates the inverse distance geometry function defined for k=2.
         *
         * @return A map of inverse distance values for the given atomic pairs.
         */
        /**
         * Calculates the distance geometry function defined for k=2.
         *
         * @param i Index of first atom.
         * @param j Index of second atom.
         * @param distances Distance matrix
         *
         * @return Inverse distance between atoms.
         */
        double k2GeomInverseDistance(const int &i, const int &j, const vector<vector<double> > &distances);
        /**
         * Weighting of 1 for all indices. Usually used for finite small
         * systems.
         *
         * @param i Index of first atom.
         * @param j Index of second atom.
         * @param distances Distance matrix
         *
         * @return Distance between atoms.
         */
        double k2GeomDistance(const int &i, const int &j, const vector<vector<double> > &distances);
        /**
         * Weighting of 1. Usually used for finite small
         * systems.
         *
         * @param i Index of first atom.
         * @param j Index of second atom.
         * @param distances Distance matrix
         *
         * @return Weight of 1.
         */
        double k2WeightUnity(const int &i, const int &j, const vector<vector<double> > &distances);
        /**
         * Weighting defined as e^(-sx), where x is the distance from
         * A->B and s is the scaling factor.
         *
         * @param i Index of first atom.
         * @param j Index of second atom.
         * @param distances Distance matrix
         * @param scale The prefactor in the exponential weighting.
         *
         * @return The exponential weight.
         */
        double k2WeightExponential(const int &i, const int &j, const vector<vector<double> > &distances, double scale);
        /**
         * Weighting defined as 1/(x^2), where x is the distance from
         * A->B.
         *
         * @param i Index of first atom.
         * @param j Index of second atom.
         * @param distances Distance matrix.
         *
         * @return The inverse of distance squared as weight.
         */
        double k2WeightSquare(const int &i, const int &j, const vector<vector<double> > &distances);
        /**
         * Calculates the cosine geometry function defined for k3.
         *
         * @param i Index of first atom.
         * @param j Index of second atom.
         * @param k Index of third atom.
         * @param distances Distance matrix
         *
         * @return The cosine value for the angle defined between the given
         * three atomic indices.
         */
        double k3GeomCosine(const int &i, const int &j, const int &k, const vector<vector<double>> &distances);
        /**
         * Calculates the angle (degrees) geometry function defined for k3.
         *
         * @param i Index of first atom.
         * @param j Index of second atom.
         * @param k Index of third atom.
         * @param distances Distance matrix
         *
         * @return The angle defined between the given three atomic indices.
         * Between 0 and 180 degrees.
         */
        double k3GeomAngle(const int &i, const int &j, const int &k, const vector<vector<double>> &distances);
        /**
         * Weighting of 1. Usually used for finite small
         * systems.
         *
         * @param i Index of first atom.
         * @param j Index of second atom.
         * @param k Index of third atom.
         * @param distances Distance matrix
         *
         * @return Weight of 1.
         */
        double k3WeightUnity(const int &i, const int &j, const int &k, const vector<vector<double>> &distances);
        /**
         * Weighting defined as e^(-sx), where x is the distance from
         * A->B->C->A and s is the scaling factor.
         *
         * @param i Index of first atom.
         * @param j Index of second atom.
         * @param k Index of third atom.
         * @param distances Distance matrix
         * @param scale The prefactor in the exponential weighting.
         *
         * @return Exponential weight.
         */
        double k3WeightExponential(const int &i, const int &j, const int &k, const vector<vector<double>> &distances, double scale);
        /**
         * Weighting defined by smooth cutoff function f(r) = 1 + y * (r/R)^(y+1)
         * - (y+1) * (r/R)^y, where r is the distance between atoms, y is
         * sharpness and R is the cutoff distance.
         *
         * @param i Index of first atom.
         * @param j Index of second atom.
         * @param k Index of third atom.
         * @param distances Distance matrix.
         * @param sharpness Sharpness of the function.
         * @param cutoff Cutoff distance of the weighting.
         *
         * @return Weight defined by smooth cutoff function.
         */
        double k3WeightSmooth(const int &i, const int &j, const int &k, const vector<vector<double> > &distances, double sharpness, double cutoff);
};

#endif