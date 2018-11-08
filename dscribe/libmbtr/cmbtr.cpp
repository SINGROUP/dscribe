#include "cmbtr.h"
#include <vector>
#include <map>
#include <tuple>
#include <math.h>
#include <string>
#include <numeric>
#include <utility>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <stdexcept>
using namespace std;

CMBTR::CMBTR(vector<vector<float> > positions, vector<int> atomicNumbers, map<int,int> atomicNumberToIndexMap, int interactionLimit, bool isLocal)
    : positions(positions)
    , atomicNumbers(atomicNumbers)
    , atomicNumberToIndexMap(atomicNumberToIndexMap)
    , interactionLimit(interactionLimit)
    , isLocal(isLocal)
    , displacementTensorInitialized(false)
    , k1IndicesInitialized(false)
    , k2IndicesInitialized(false)
    , k3IndicesInitialized(false)
    , k1MapInitialized(false)
    , k2MapInitialized(false)
    , k3MapInitialized(false)
{
}

vector<vector<vector<float> > > CMBTR::getDisplacementTensor()
{
    // Use cached value if possible
    if (!this->displacementTensorInitialized) {
        int nAtoms = this->atomicNumbers.size();

        // Initialize tensor
        vector<vector<vector<float> > > tensor(nAtoms, vector<vector<float> >(nAtoms, vector<float>(3)));

        // Calculate the distance between all pairs and store in the tensor
        for (int i=0; i < nAtoms; ++i) {
            for (int j=0; j < nAtoms; ++j) {

                // Due to symmetry only upper triangular part is processed.
                if (i <= j) {
                    continue;
                }

                // Calculate distance between the two atoms, store in tensor
                vector<float>& iPos = this->positions[i];
                vector<float>& jPos = this->positions[j];
                vector<float> diff(3);
                vector<float> negDiff(3);

                for (int k=0; k < 3; ++k) {
                    float iDiff = jPos[k] - iPos[k];
                    diff[k] = iDiff;
                    negDiff[k] = -iDiff;
                }

                tensor[i][j] = diff;
                tensor[j][i] = negDiff;
            }
        }
        this->displacementTensor = tensor;
        this->displacementTensorInitialized = true;
    }
    return this->displacementTensor;
}
vector<vector<float> > CMBTR::getDistanceMatrix()
{
    int nAtoms = this->atomicNumbers.size();

    // Initialize matrix
    vector<vector<float> > distanceMatrix(nAtoms, vector<float>(nAtoms));

    // Displacement tensor
    vector<vector<vector<float> > > tensor = this->getDisplacementTensor();

    // Calculate distances
    for (int i=0; i < nAtoms; ++i) {
        for (int j=0; j < nAtoms; ++j) {

            // Due to symmetry only upper triangular part is processed.
            if (i <= j) {
                continue;
            }

            float norm = 0;
            vector<float>& iPos = tensor[i][j];
            for (int k=0; k < 3; ++k) {
                norm += pow(iPos[k], 2.0);
            }
            norm = sqrt(norm);
            distanceMatrix[i][j] = norm;
            distanceMatrix[j][i] = norm;
        }
    }

    return distanceMatrix;
}

vector<index1d> CMBTR::getk1Indices()
{
    // Use cached value if possible
    if (!this->k1IndicesInitialized) {

        int nAtoms = this->atomicNumbers.size();

        // Initialize the map containing the mappings
        vector<index1d> indexList;

        for (int i=0; i < nAtoms; ++i) {
            // Only consider triplets that have one atom in the original
            // cell
            if (i < this->interactionLimit) {
                index1d key = {i};
                indexList.push_back(key);
            }
        }
        this->k1Indices = indexList;
        this->k1IndicesInitialized = true;
    }
    return this->k1Indices;
}

vector<index2d> CMBTR::getk2Indices()
{
    // Use cached value if possible
    if (!this->k2IndicesInitialized) {

        int nAtoms = this->atomicNumbers.size();

        // Initialize the map containing the mappings
        vector<index2d> indexList;

        for (int i=0; i < nAtoms; ++i) {
            for (int j=0; j < nAtoms; ++j) {

                // Due to symmetry only upper triangular part is processed.
                if (j > i) {

                    // Only consider triplets that have one atom in the original
                    // cell
                    if (i < this->interactionLimit || j < this->interactionLimit) {
                        index2d key = {i, j};
                        indexList.push_back(key);
                    }
                }
            }
        }
        this->k2Indices = indexList;
        this->k2IndicesInitialized = true;
    }
    return this->k2Indices;
}

vector<index3d> CMBTR::getk3Indices()
{
    // Use cached value if possible
    if (!this->k3IndicesInitialized) {

        int nAtoms = this->atomicNumbers.size();

        // Initialize the map containing the mappings
        vector<index3d> indexList;

        for (int i=0; i < nAtoms; ++i) {
            for (int j=0; j < nAtoms; ++j) {
                for (int k=0; k < nAtoms; ++k) {

                    // Only consider triplets that have one atom in the original
                    // cell
                    if (i < this->interactionLimit || j < this->interactionLimit || k < this->interactionLimit) {
                        // Calculate angle for all index permutations from choosing
                        // three out of nAtoms. The same atom cannot be present twice
                        // in the permutation.
                        if (j != i && k != j && k != i) {
                            // The angles are symmetric: ijk = kji. The value is
                            // calculated only for the triplet where k > i.
                            if (k > i) {
                                index3d key = {i, j, k};
                                indexList.push_back(key);
                            }
                        }
                    }
                }
            }
        }
        this->k3Indices = indexList;
        this->k3IndicesInitialized = true;
    }
    return this->k3Indices;
}

map<index1d, float> CMBTR::k1GeomAtomicNumber(const vector<index1d> &indexList)
{
    map<index1d,float> valueMap;
    for (const index1d& index : indexList) {
        int i = index.i;
        int atomicNumber = this->atomicNumbers[i];
        valueMap[index] = atomicNumber;
    }

    return valueMap;
}

map<index2d, float> CMBTR::k2GeomInverseDistance(const vector<index2d> &indexList)
{
    vector<vector<float> > distMatrix = this->getDistanceMatrix();

    map<index2d,float> valueMap;
    for (const index2d& index : indexList) {
        int i = index.i;
        int j = index.j;

        float invDist = 1/distMatrix[i][j];
        valueMap[index] = invDist;
    }

    return valueMap;
}

map<index3d, float> CMBTR::k3GeomCosine(const vector<index3d> &indexList)
{
    vector<vector<float> > distMatrix = this->getDistanceMatrix();
    vector<vector<vector<float> > > dispTensor = this->getDisplacementTensor();

    map<index3d,float> valueMap;
    for (const index3d& index : indexList) {
        int i = index.i;
        int j = index.j;
        int k = index.k;

        vector<float> a = dispTensor[i][j];
        vector<float> b = dispTensor[k][j];
        float dotProd = inner_product(a.begin(), a.end(), b.begin(), 0.0);
        float cosine = dotProd / (distMatrix[i][j]*distMatrix[k][j]);
        valueMap[index] = cosine;
    }

    return valueMap;
}

map<index1d, float> CMBTR::k1WeightUnity(const vector<index1d> &indexList)
{
    map<index1d, float> valueMap;

    for (const index1d& index : indexList) {
        valueMap[index] = 1;
    }

    return valueMap;
}

map<index2d, float> CMBTR::k2WeightUnity(const vector<index2d> &indexList)
{
    map<index2d, float> valueMap;

    for (const index2d& index : indexList) {
        valueMap[index] = 1;
    }

    return valueMap;
}

map<index3d, float> CMBTR::k3WeightUnity(const vector<index3d> &indexList)
{
    map<index3d, float> valueMap;

    for (const index3d& index : indexList) {
        valueMap[index] = 1;
    }

    return valueMap;
}

map<index2d, float> CMBTR::k2WeightExponential(const vector<index2d> &indexList, float scale, float cutoff)
{
    vector<vector<float> > distMatrix = this->getDistanceMatrix();
    map<index2d, float> valueMap;

    for (const index2d& index : indexList) {
        int i = index.i;
        int j = index.j;

        float dist = distMatrix[i][j];
        float expValue = exp(-scale*dist);
        if (expValue >= cutoff) {
            valueMap[index] = expValue;
        }
    }

    return valueMap;
}

map<index3d, float> CMBTR::k3WeightExponential(const vector<index3d> &indexList, float scale, float cutoff)
{
    vector<vector<float> > distMatrix = this->getDistanceMatrix();
    map<index3d, float> valueMap;

    for (const index3d& index : indexList) {
        int i = index.i;
        int j = index.j;
        int k = index.k;

        float dist1 = distMatrix[i][j];
        float dist2 = distMatrix[j][k];
        float dist3 = distMatrix[k][i];
        float distTotal = dist1 + dist2 + dist3;
        float expValue = exp(-scale*distTotal);
        if (expValue >= cutoff) {
            valueMap[index] = expValue;
        }
    }

    return valueMap;
}

pair<map<index1d, vector<float> >, map<index1d,vector<float> > > CMBTR::getK1GeomsAndWeights(string geomFunc, string weightFunc, map<string, float> parameters)
{
    // Use cached value if possible
    if (!this->k1IndicesInitialized) {

        vector<index1d> indexList = this->getk1Indices();

        // Initialize the maps
        map<index1d, vector<float> > geomMap;
        map<index1d, vector<float> > weightMap;

        // Calculate all geometry values
        map<index1d, float> geomValues;
        if (geomFunc == "atomic_number") {
            geomValues = this->k1GeomAtomicNumber(indexList);
        } else {
            throw invalid_argument("Invalid geometry function.");
        }

        // Calculate all weighting values
        map<index1d, float> weightValues;
        if (weightFunc == "unity") {
            weightValues = this->k1WeightUnity(indexList);
        } else {
            throw invalid_argument("Invalid weighting function.");
        }

        // Save the geometry and distances to the maps
        for (index1d& index : indexList) {
            int i = index.i;

            // By default C++ map will return a default constructed float when
            // if the key is not found with the []-operator. Instead we
            // explicitly ask if the key is present.
            if ( weightValues.find(index) == weightValues.end() ) {
                continue;
            }

            float geomValue = geomValues[index];
            float weightValue = weightValues[index];

            // Get the index of the present elements in the final vector
            int i_elem = this->atomicNumbers[i];
            int i_index = this->atomicNumberToIndexMap[i_elem];

            // Save information in the part where j_index >= i_index
            index1d indexKey = {i_index};

            // Save the values
            geomMap[indexKey].push_back(geomValue);
            weightMap[indexKey].push_back(weightValue);
        }

        this->k1Map = make_pair(geomMap, weightMap);
        this->k1MapInitialized = true;
    }
    return this->k1Map;
}

pair<map<index2d, vector<float> >, map<index2d,vector<float> > > CMBTR::getK2GeomsAndWeights(string geomFunc, string weightFunc, map<string, float> parameters)
{
    // Use cached value if possible
    if (!this->k2IndicesInitialized) {

        vector<index2d> indexList = this->getk2Indices();

        // Initialize the maps
        map<index2d, vector<float> > geomMap;
        map<index2d, vector<float> > weightMap;

        // Calculate all geometry values
        map<index2d, float> geomValues;
        if (geomFunc == "inverse_distance") {
            geomValues = this->k2GeomInverseDistance(indexList);
        } else {
            throw invalid_argument("Invalid geometry function.");
        }

        // Calculate all weighting values
        map<index2d, float> weightValues;
        if (weightFunc == "exponential") {
            float scale = parameters["scale"];
            float cutoff = parameters["cutoff"];
            weightValues = this->k2WeightExponential(indexList, scale, cutoff);
        } else if (weightFunc == "unity") {
            weightValues = this->k2WeightUnity(indexList);
        } else {
            throw invalid_argument("Invalid weighting function.");
        }

        // Save the geometry and distances to the maps
        for (index2d& index : indexList) {
            int i = index.i;
            int j = index.j;

            // By default C++ map will return a default constructed float when
            // if the key is not found with the []-operator. Instead we
            // explicitly ask if the key is present.
            if ( weightValues.find(index) == weightValues.end() ) {
                continue;
            }

            float geomValue = geomValues[index];
            float weightValue = weightValues[index];

            // When the pair of atoms are in different copies of the cell, the
            // weight is halved. This is done in order to avoid double counting
            // the same distance in the opposite direction. This correction
            // makes periodic cells with different translations equal and also
            // supercells equal to the primitive cell within a constant that is
            // given by the number of repetitions of the primitive cell in the
            // supercell.
            if (!this->isLocal) {
                if (!((i < this->interactionLimit) && (j < this->interactionLimit))) {
                    weightValue /= 2;
                }
            }

            // Get the index of the present elements in the final vector
            int i_elem = this->atomicNumbers[i];
            int j_elem = this->atomicNumbers[j];
            int i_index = this->atomicNumberToIndexMap[i_elem];
            int j_index = this->atomicNumberToIndexMap[j_elem];

            // Save information in the part where j_index >= i_index
            index2d indexKey;
            if (j_index < i_index) {
                indexKey = {j_index, i_index};
            } else {
                indexKey = {i_index, j_index};
            }

            // Save the values
            geomMap[indexKey].push_back(geomValue);
            weightMap[indexKey].push_back(weightValue);
        }

        this->k2Map = make_pair(geomMap, weightMap);
        this->k2MapInitialized = true;
    }
    return this->k2Map;
}

pair<map<index3d, vector<float> >, map<index3d,vector<float> > > CMBTR::getK3GeomsAndWeights(string geomFunc, string weightFunc, map<string, float> parameters)
{
    // Use cached value if possible
    if (!this->k3IndicesInitialized) {

        vector<index3d> indexList = this->getk3Indices();

        // Initialize the maps
        map<index3d, vector<float> > geomMap;
        map<index3d, vector<float> > weightMap;

        // Calculate all geometry values
        map<index3d, float> geomValues;
        if (geomFunc == "cosine") {
            geomValues = this->k3GeomCosine(indexList);
        } else {
            throw invalid_argument("Invalid geometry function.");
        }

        // Calculate all weighting values
        map<index3d, float> weightValues;
        if (weightFunc == "exponential") {
            float scale = parameters["scale"];
            float cutoff = parameters["cutoff"];
            weightValues = this->k3WeightExponential(indexList, scale, cutoff);
        } else if (weightFunc == "unity") {
            weightValues = this->k3WeightUnity(indexList);
        } else {
            throw invalid_argument("Invalid weighting function.");
        }

        // Save the geometry and distances to the maps
        for (index3d& index : indexList) {
            int i = index.i;
            int j = index.j;
            int k = index.k;

            // By default C++ map will return a default constructed float when
            // if the key is not found with the []-operator. Instead we
            // explicitly ask if the key is present.
            if ( weightValues.find(index) == weightValues.end() ) {
                continue;
            }

            float geomValue = geomValues[index];
            float weightValue = weightValues[index];

            // When at least one of the atoms is in a different copy of the cell, the
            // weight is halved. This is done in order to avoid double counting
            // the same distance in the opposite direction. This correction
            // makes periodic cells with different translations equal and also
            // supercells equal to the primitive cell within a constant that is
            // given by the number of repetitions of the primitive cell in the
            // supercell.
            if (!this->isLocal) {
                if (!((i < this->interactionLimit) && (j < this->interactionLimit) && (k < this->interactionLimit))) {
                    weightValue /= 2;
                }
            }

            // Get the index of the present elements in the final vector
            int i_elem = this->atomicNumbers[i];
            int j_elem = this->atomicNumbers[j];
            int k_elem = this->atomicNumbers[k];
            int i_index = this->atomicNumberToIndexMap[i_elem];
            int j_index = this->atomicNumberToIndexMap[j_elem];
            int k_index = this->atomicNumberToIndexMap[k_elem];

            // Save information in the part where k_index >= i_index
            index3d indexKey;
            if (k_index < i_index) {
                indexKey = {k_index, j_index, i_index};
            } else {
                indexKey = {i_index, j_index, k_index};
            }

            // Save the values
            geomMap[indexKey].push_back(geomValue);
            weightMap[indexKey].push_back(weightValue);
        }

        this->k3Map = make_pair(geomMap, weightMap);
        this->k3MapInitialized = true;
    }
    return this->k3Map;
}

pair<map<string,vector<float> >, map<string,vector<float> > > CMBTR::getK1GeomsAndWeightsCython(string geomFunc, string weightFunc, map<string, float> parameters)
{
    pair<map<index1d,vector<float> >, map<index1d,vector<float> > > cMap = this->getK1GeomsAndWeights(geomFunc, weightFunc, parameters);
    map<index1d, vector<float> > geomValues = cMap.first;
    map<index1d, vector<float> > weightValues = cMap.second;

    map<string, vector<float> > cythonGeom;
    map<string, vector<float> > cythonWeight;

    for (auto const& x : geomValues) {
        stringstream ss;
        ss << x.first.i;
        string stringKey = ss.str();
        cythonGeom[stringKey] = x.second;
        cythonWeight[stringKey] = weightValues[x.first];
    }
    return make_pair(cythonGeom, cythonWeight);
}

pair<map<string,vector<float> >, map<string,vector<float> > > CMBTR::getK2GeomsAndWeightsCython(string geomFunc, string weightFunc, map<string, float> parameters)
{
    pair<map<index2d,vector<float> >, map<index2d,vector<float> > > cMap = this->getK2GeomsAndWeights(geomFunc, weightFunc, parameters);
    map<index2d, vector<float> > geomValues = cMap.first;
    map<index2d, vector<float> > weightValues = cMap.second;

    map<string, vector<float> > cythonGeom;
    map<string, vector<float> > cythonWeight;

    for (auto const& x : geomValues) {
        stringstream ss;
        ss << x.first.i;
        ss << ",";
        ss << x.first.j;
        string stringKey = ss.str();
        cythonGeom[stringKey] = x.second;
        cythonWeight[stringKey] = weightValues[x.first];
    }
    return make_pair(cythonGeom, cythonWeight);
}

pair<map<string,vector<float> >, map<string,vector<float> > > CMBTR::getK3GeomsAndWeightsCython(string geomFunc, string weightFunc, map<string, float> parameters)
{
    pair<map<index3d,vector<float> >, map<index3d,vector<float> > > cMap = this->getK3GeomsAndWeights(geomFunc, weightFunc, parameters);
    map<index3d, vector<float> > geomValues = cMap.first;
    map<index3d, vector<float> > weightValues = cMap.second;

    map<string, vector<float> > cythonGeom;
    map<string, vector<float> > cythonWeight;

    for (auto const& x : geomValues) {
        stringstream ss;
        ss << x.first.i;
        ss << ",";
        ss << x.first.j;
        ss << ",";
        ss << x.first.k;
        string stringKey = ss.str();
        cythonGeom[stringKey] = x.second;
        cythonWeight[stringKey] = weightValues[x.first];
    }
    return make_pair(cythonGeom, cythonWeight);
}
