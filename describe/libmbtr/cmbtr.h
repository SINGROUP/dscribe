#ifndef CMBTR_H
#define CMBTR_H

#include <vector>
#include <map>
#include <utility>
using namespace std;

/**
 * Implementation for the performance-critical parts of MBTR.
 */
class CMBTR {

    public:
        /**
         * Constructor
         *
         * @param positions Atomic positions in cartesian coordinates.
         * @param atomicNumbers Atomic numbers.
         * @param atomicNumberToIndexMap Mapping between atomic numbers and
         * their position in the final MBTR vector.
         * @param cellLimit The number of atoms in the original cell. The atoms
         * from 0-cellLimit belong to the original simulation cell, rest are
         * periodic copies.
         */
        CMBTR(vector<vector<float> > positions, vector<int> atomicNumbers, map<int,int> atomicNumberToIndexMap, int cellLimit);

        /**
         * Calculates a 3D matrix of distance vectors between atomic positions.
         *
         * @return A 3D matrix of displacement vectors. First index is the
         * index of ith atom, second index is the index of jth atom, third
         * index is the cartesian component.
         */
        vector<vector<vector<float> > > getDisplacementTensor();

        /**
         * Calculates a 2D matrix of distances between atomic positions.
         *
         * @return A 2D matrix of distances. First index is the
         * index of ith atom, second index is the index of jth atom.
         */
        vector<vector<float> > getDistanceMatrix();

        /**
         * Calculates a 2D matrix of inverse distances between atomic
         * positions.
         *
         * @return A 2D matrix of inverse distances. First index is the
         * index of ith atom, second index is the index of jth atom.
         */
        vector<vector<float> > getInverseDistanceMatrix();

        /**
         * Calculates a mapping of inverse distances between two different
         * atomic elements.
         *
         * @return A map where the key is a pair of atomic numbers and the
         * value is a list of inverse distances corresponding to pairs of atoms
         * with the given atomic numbers.
         */
        map<pair<int, int>, vector<float> > getInverseDistanceMap();

        /**
         * Calculates a 3D matrix of cosines of angles between triplets of
         * atoms. The cosines are returned as a map that maps triplets of atomic
         * indices to an angle. This way we save half of the space that would
         * be needed it the angles were stored in a regular 3D array where
         * elements in ijk are identical to kji.
         *
         * @return A 3D matrix of angles. First index is the ith
         * atom, second index is the jth atom and third index is the kth atom.
         * The jth atom is the central atom in forming the angle.
         */
        map<vector<int>, float> getAngleCosines();

    private:
        vector<vector<float> > positions;
        vector<int> atomicNumbers;
        map<int,int> atomicNumberToIndexMap;
        int cellLimit;
        vector<vector<vector<float> > > displacementTensor;
        bool displacementTensorInitialized;
};

#endif
