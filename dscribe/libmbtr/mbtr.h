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
         * Default constructor
         */
        MBTR() {};

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
         * @param cellIndices
         * @param local Whether a local or a global MBTR is calculated.
         */
        MBTR(vector<vector<float> > positions, vector<int> atomicNumbers, map<int,int> atomicNumberToIndexMap, int interactionLimit,  vector<vector<int>> cellIndices, bool local=false);
        map<string, vector<float> > getK1(string geomFunc, string weightFunc, map<string, float> parameters, float min, float max, float sigma, float n);
        map<string, vector<float> > getK2(const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters, float min, float max, float sigma, float n);
        map<string, vector<float> > getK3(const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters, float min, float max, float sigma, float n);
        vector<float> gaussian(const float &center, const float &weight, const float &start, const float &dx, const float &sigmasqrt2, const int &n);

        /**
         * Returns a list of 1D indices for the atom combinations that need to
         * be calculated for the k=1 term.
         *
         * @return A list of 1D indices.
         */
        vector<index1d> getk1Indices();

        /**
         * Returns a list of 2D indices for the atom combinations that need to
         * be calculated for the k=2 term.
         *
         * @return A list of 2D indices.
         */
        vector<index2d> getk2Indices(const vector<vector<int> > &neighbours);

        /**
         * Returns a list of 3D indices for the atom combinations that need to
         * be calculated for the k=3 term.
         *
         * @return A list of 3D indices for k3.
         */
        vector<index3d> getk3Indices(const vector<vector<int> > &neighbours);

        /**
         * Weighting of 1 for all indices. Usually used for finite small
         * systems.
         *
         * @return
         */
        map<index1d, float> k1WeightUnity(const vector<index1d> &indexList);

        /**
         * Weighting of 1 for all indices. Usually used for finite small
         * systems.
         *
         * @return
         */
        map<index2d, float> k2WeightUnity(const vector<index2d> &indexList);

        /**
         * Weighting of 1 for all indices. Usually used for finite small
         * systems.
         *
         * @return
         */
        map<index3d, float> k3WeightUnity(const vector<index3d> &indexList);

        /**
         * Weighting defined as e^(-sx), where x is the distance from
         * A->B and s is the scaling factor.
         *
         * @param indexList List of pairs of atomic indices.
         * @param scale The prefactor in the exponential weighting.
         * @param cutoff Minimum value of the weighting to consider.
         * @return Map of distances for pairs of atomic indices.
         */
        map<index2d, float> k2WeightExponential(const vector<index2d> &indexList, float scale, float cutoff, const vector<vector<float> > &distances);

        /**
         * Weighting defined as e^(-sx), where x is the distance from
         * A->B->C->A and s is the scaling factor.
         *
         * @param indexList List of triplets of atomic indices.
         * @param scale The prefactor in the exponential weighting.
         * @param cutoff Minimum value of the weighting to consider.
         * @return Map of distances for triplets of atomic indices.
         */
        map<index3d, float> k3WeightExponential(const vector<index3d> &indexList, float scale, float cutoff, const vector<vector<float> > &distances);

        /**
         * Calculates the geometry function based on atomic numbers defined for
         * k=1.
         *
         * @return A map of atomic numbers for the given indices.
         */
        map<index1d, float> k1GeomAtomicNumber(const vector<index1d> &indexList);

        /**
         * Calculates the distance geometry function defined for k=2.
         *
         * @return A map of distance values for the given atomic pairs.
         */
        map<index2d, float> k2GeomDistance(const vector<index2d> &indexList, const vector<vector<float> > &distances);

        /**
         * Calculates the inverse distance geometry function defined for k=2.
         *
         * @return A map of inverse distance values for the given atomic pairs.
         */
        map<index2d, float> k2GeomInverseDistance(const vector<index2d> &indexList, const vector<vector<float> > &distances);

        /**
         * Calculates the angle (degrees) geometry function defined for k3.
         *
         * @return The angle defined between the given three atomic indices.
         * Between 0 and 180 degrees.
         */
        map<index3d, float> k3GeomAngle(const vector<index3d> &indexList, const vector<vector<float> > &distances);

        /**
         * Calculates the cosine geometry function defined for k3.
         *
         * @return The cosine value for the angle defined between the given
         * three atomic indices.
         */
        map<index3d, float> k3GeomCosine(const vector<index3d> &indexList, const vector<vector<float> > &distances);

        /**
         * Calculates a 3D matrix of distance vectors between atomic positions.
         *
         * @return A 3D matrix of displacement vectors. First index is the
         * index of ith atom, second index is the index of jth atom, third
         * index is the cartesian component.
         */
        //vector<vector<vector<float> > > getDisplacementTensor();

        /**
         * Calculates a 2D matrix of distances between atomic positions.
         *
         * @return A 2D matrix of distances. First index is the
         * index of ith atom, second index is the index of jth atom.
         */
        //vector<vector<float> > getDistanceMatrix();

        /**
         * Calculates a list of values for the k=1 geometry functions and the
         * corresponding weights for each pair of atomic numbers.
         *
         * @return A pair of maps for the geometry- and weighting function
         * values for each pair of atomic elements.
         */
        pair<map<index1d,vector<float> >, map<index1d,vector<float> > > getK1GeomsAndWeights(string geomFunc, string weightFunc, map<string, float> parameters=map<string, float>());

        /**
         * Calculates a list of values for the k=2 geometry functions and the
         * corresponding weights for each pair of atomic numbers.
         *
         * @return A pair of maps for the geometry- and weighting function
         * values for each pair of atomic elements.
         */
        pair<map<index2d,vector<float> >, map<index2d,vector<float> > > getK2GeomsAndWeights(const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters=map<string, float>());

        /**
         * Calculates a list of values for the k=3 geometry functions and the
         * corresponding weights for each triplet of atomic numbers.
         *
         * @return A pair of maps for the geometry- and weighting function
         * values for each triplet of atomic elements.
         */
        pair<map<index3d,vector<float> >, map<index3d,vector<float> > > getK3GeomsAndWeights(const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters=map<string, float>());

        /**
         * A convenience function that provides a Cython-compatible interface
         * to the getK1GeomsAndWeights-function. Cython cannot handle custom
         * map keys, so a string key is provided by this function. The key is
         * formed by casting the atomic index to a string.
         */
        pair<map<string,vector<float> >, map<string,vector<float> > > getK1GeomsAndWeightsCython(string geomFunc, string weightFunc, map<string, float> parameters=map<string, float>());

        /**
         * A convenience function that provides a Cython-compatible interface
         * to the getK2GeomsAndWeights-function. Cython cannot handle custom
         * map keys, so a string key is provided by this function. The key is
         * formed by casting the atomic indices to strings and concatenating
         * them with comma as a separator.
         */
        pair<map<string,vector<float> >, map<string,vector<float> > > getK2GeomsAndWeightsCython(const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters=map<string, float>());

        /**
         * A convenience function that provides a Cython-compatible interface
         * to the getK3GeomsAndWeights-function. Cython cannot handle custom
         * map keys, so a string key is provided by this function. The key is
         * formed by casting the atomic indices to strings and concatenating
         * them with comma as a separator.
         */
        pair<map<string,vector<float> >, map<string,vector<float> > > getK3GeomsAndWeightsCython(const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters=map<string, float>());

    private:
        vector<vector<float> > positions;
        vector<int> atomicNumbers;
        map<int,int> atomicNumberToIndexMap;
        int interactionLimit;
        vector<vector<int> > cellIndices;
        bool isLocal;
        vector<vector<vector<float> > > displacementTensor;
        bool displacementTensorInitialized;
        vector<vector<float> > distanceMatrix;
        bool distanceMatrixInitialized;
        vector<index1d> k1Indices;
        bool k1IndicesInitialized;
        vector<index2d> k2Indices;
        bool k2IndicesInitialized;
        vector<index3d> k3Indices;
        bool k3IndicesInitialized;
        pair<map<index1d, vector<float> >, map<index1d, vector<float> > > k1Map;
        bool k1MapInitialized;
        pair<map<index2d, vector<float> >, map<index2d, vector<float> > > k2Map;
        bool k2MapInitialized;
        pair<map<index3d, vector<float> >, map<index3d, vector<float> > > k3Map;
        bool k3MapInitialized;

        float k1GeomAtomicNumber(const int &i);
        float k1WeightUnity(const int &i);
        float k2GeomInverseDistance(const int &i, const int &j, const vector<vector<float> > &distances);
        float k2GeomDistance(const int &i, const int &j, const vector<vector<float> > &distances);
        float k2WeightUnity(const int &i, const int &j, const vector<vector<float> > &distances);
        float k2WeightExponential(const int &i, const int &j, const vector<vector<float> > &distances, const float &scale);
        float k3GeomCosine(const int &i, const int &j, const int &k, const vector<vector<float> > &distances);
        float k3GeomAngle(const int &i, const int &j, const int &k, const vector<vector<float> > &distances);
        float k3WeightUnity(const int &i, const int &j, const int &k, const vector<vector<float> > &distances);
        float k3WeightExponential(const int &i, const int &j, const int &k, const vector<vector<float> > &distances, const float &scale);
};

#endif
