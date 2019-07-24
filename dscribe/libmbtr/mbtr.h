#ifndef MBTR_H
#define MBTR_H

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
        map<string, vector<float>> getK1(const vector<int> &Z, const string &geomFunc, const string &weightFunc, const map<string, float> &parameters, float min, float max, float sigma, int n);
        map<string, vector<float>> getK2(const vector<int> &Z, const vector<vector<float>> &distances, const vector<vector<int>> &neighbours, const string &geomFunc, const string &weightFunc, const map<string, float> &parameters, float min, float max, float sigma, int n);
        map<string, vector<float>> getK3(const vector<int> &Z, const vector<vector<float>> &distances, const vector<vector<int>> &neighbours, const string &geomFunc, const string &weightFunc, const map<string, float> &parameters, float min, float max, float sigma, int n);
        vector<map<string, vector<float>>> getK2Local(const vector<int> &indices, const vector<int> &Z, const vector<vector<float>> &distances, const vector<vector<int>> &neighbours, const string &geomFunc, const string &weightFunc, const map<string, float> &parameters, float min, float max, float sigma, int n);
        vector<map<string, vector<float>>> getK3Local(const vector<int> &indices, const vector<int> &Z, const vector<vector<float>> &distances, const vector<vector<int>> &neighbours, const string &geomFunc, const string &weightFunc, const map<string, float> &parameters, float min, float max, float sigma, int n);
        vector<float> gaussian(float center, float weight, float start, float dx, float sigmasqrt2, int n);


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
        float k1GeomAtomicNumber(const int &i, const vector<int> &Z);

        /**
         * Weighting of 1. Usually used for finite small
         * systems.
         *
         * @param i Index of first atom.
         *
         * @return Weight of 1.
         */
        float k1WeightUnity(const int &i);
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
        float k2GeomInverseDistance(const int &i, const int &j, const vector<vector<float> > &distances);
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
        float k2GeomDistance(const int &i, const int &j, const vector<vector<float> > &distances);
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
        float k2WeightUnity(const int &i, const int &j, const vector<vector<float> > &distances);
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
        float k2WeightExponential(const int &i, const int &j, const vector<vector<float> > &distances, float scale);
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
        float k3GeomCosine(const int &i, const int &j, const int &k, const vector<vector<float>> &distances);
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
        float k3GeomAngle(const int &i, const int &j, const int &k, const vector<vector<float>> &distances);
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
        float k3WeightUnity(const int &i, const int &j, const int &k, const vector<vector<float>> &distances);
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
        float k3WeightExponential(const int &i, const int &j, const int &k, const vector<vector<float>> &distances, float scale);
};

#endif
