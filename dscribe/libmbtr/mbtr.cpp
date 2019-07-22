#include "mbtr.h"
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
using namespace std;

MBTR::MBTR(vector<vector<float> > positions, vector<int> atomicNumbers, map<int,int> atomicNumberToIndexMap, int interactionLimit, vector<vector<int> > cellIndices, bool isLocal)
    : positions(positions)
    , atomicNumbers(atomicNumbers)
    , atomicNumberToIndexMap(atomicNumberToIndexMap)
    , interactionLimit(interactionLimit)
    , cellIndices(cellIndices)
    , isLocal(isLocal)
    , displacementTensorInitialized(false)
    , distanceMatrixInitialized(false)
    , k1IndicesInitialized(false)
    , k2IndicesInitialized(false)
    , k3IndicesInitialized(false)
    , k1MapInitialized(false)
    , k2MapInitialized(false)
    , k3MapInitialized(false)
{
}

map<string, vector<double> > MBTR::getK1(string geomFunc, string weightFunc, map<string, float> parameters, float min, float max, float sigma, float n)
{
    map<string, vector<double> > k1Map;
    int nAtoms = this->atomicNumbers.size();

    for (int i=0; i < nAtoms; ++i) {
        // Only consider atoms within the original cell
        if (i < this->interactionLimit) {

            // Calculate all geometry values
            float geom;
            if (geomFunc == "atomic_number") {
                geom = k1GeomAtomicNumber(i);
            } else {
                throw invalid_argument("Invalid geometry function.");
            }

            // Calculate all weighting values
            float weight;
            if (weightFunc == "unity") {
                weight = k1WeightUnity(i);
            } else {
                throw invalid_argument("Invalid weighting function.");
            }

            // Calculate gaussian
            vector<double> gauss = gaussian(geom, sigma, weight, min, max, n);

            // Get the index of the present elements in the final vector
            int i_elem = this->atomicNumbers[i];
            int i_index = this->atomicNumberToIndexMap[i_elem];

            // Form the key as string to enable passing it through cython
            string stringKey = to_string(i_index);

            // Sum gaussian into output
            try {
                vector<double> &old = k1Map.at(stringKey);
                transform(old.begin(), old.end(), gauss.begin(), old.begin(), plus<double>());
            } catch(const out_of_range& oor) {
                k1Map[stringKey] = gauss;
            }
        }
    }
    return k1Map;
}

map<string, vector<double> > MBTR::getK2(const vector<vector<double> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters, float min, float max, float sigma, float n)
{
    map<string, vector<double> > k2Map;
    int nAtoms = this->atomicNumbers.size();

    // We have to loop over all atoms in the system
    for (int i=0; i < nAtoms; ++i) {

        // For each atom we loop only over the neighbours
        const vector<int> &i_neighbours = neighbours[i];
        for (const int &j : i_neighbours) {
            if (j > i) {

                // Only consider pairs that have one atom in the original
                // cell
                if (i < this->interactionLimit || j < this->interactionLimit) {

                    // Calculate geom function TODO: remove if statement here
                    float geom;
                    if (geomFunc == "inverse_distance") {
                        geom = k2GeomInverseDistance(i, j, distances);
                    } else if (geomFunc == "distance") {
                        geom = k2GeomDistance(i, j, distances);
                    } else {
                        throw invalid_argument("Invalid geometry function.");
                    }

                    // Calculate weight
                    float weight;
                    if (weightFunc == "exponential" || weightFunc == "exp") {
                        float scale = parameters["scale"];
                        float cutoff = parameters["cutoff"];
                        weight = k2WeightExponential(i, j, distances, scale, cutoff);
                    } else if (weightFunc == "unity") {
                        weight = k2WeightUnity(i, j, distances);
                    } else {
                        throw invalid_argument("Invalid weighting function.");
                    }

                    // Ignore stuff with no weight
                    if (weight == 0.0) {
                        continue;
                    }

                    // Calculate gaussian
                    vector<double> gauss = gaussian(geom, sigma, weight, min, max, n);

                    // Find position in output
                    // When the pair of atoms are in different copies of the cell, the
                    // weight is halved. This is done in order to avoid double counting
                    // the same distance in the opposite direction. This correction
                    // makes periodic cells with different translations equal and also
                    // supercells equal to the primitive cell within a constant that is
                    // given by the number of repetitions of the primitive cell in the
                    // supercell.
                    if (!this->isLocal) {
                        vector<int> i_copy = this->cellIndices[i];
                        vector<int> j_copy = this->cellIndices[j];

                        if (i_copy != j_copy) {
                            weight /= 2;
                        }
                    }

                    // Get the index of the present elements in the final vector
                    int i_elem = this->atomicNumbers[i];
                    int j_elem = this->atomicNumbers[j];
                    int i_index = this->atomicNumberToIndexMap[i_elem];
                    int j_index = this->atomicNumberToIndexMap[j_elem];

                    // Save information in the part where j_index >= i_index
                    if (j_index < i_index) {
                        int temp = j_index;
                        j_index = i_index;
                        i_index = temp;
                    }

                    // Form the key as string to enable passing it through cython
                    stringstream ss;
                    ss << i_index;
                    ss << ",";
                    ss << j_index;
                    string stringKey = ss.str();

                    // Sum gaussian into output
                    try {
                        vector<double> &old = k2Map.at(stringKey);
                        transform(old.begin(), old.end(), gauss.begin(), old.begin(), plus<double>());
                    } catch(const out_of_range& oor) {
                        k2Map[stringKey] = gauss;
                    }
                }
            }
        }
    }
    return k2Map;
}

map<string, vector<double> > MBTR::getK3(const vector<vector<double> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters, float min, float max, float sigma, float n)
{
    map<string, vector<double> > k3Map;
    int nAtoms = this->atomicNumbers.size();

    for (int i=0; i < nAtoms; ++i) {

        // For each atom we loop only over the atoms triplest that are
        // within the neighbourhood
        const vector<int> &i_neighbours = neighbours[i];
        for (const int &j : i_neighbours) {
            for (const int &k : i_neighbours) {
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

                            // Calculate all geometry values
                            float geom;
                            if (geomFunc == "cosine") {
                                geom = k3GeomCosine(i, j, k, distances);
                            } else if (geomFunc == "angle") {
                                geom = k3GeomAngle(i, j, k, distances);
                            } else {
                                throw invalid_argument("Invalid geometry function.");
                            }

                            // Calculate all weighting values
                            float weight;
                            if (weightFunc == "exponential" || weightFunc == "exp") {
                                float scale = parameters["scale"];
                                float cutoff = parameters["cutoff"];
                                weight = k3WeightExponential(i, j, k, distances, scale, cutoff);
                            } else if (weightFunc == "unity") {
                                weight = k3WeightUnity(i, j, k, distances);
                            } else {
                                throw invalid_argument("Invalid weighting function.");
                            }

                            // Ignore stuff with no weight
                            if (weight == 0.0) {
                                continue;
                            }

                            // Calculate gaussian
                            vector<double> gauss = gaussian(geom, sigma, weight, min, max, n);

                            // The contributions are weighted by their multiplicity arising from
                            // the translational symmetry. Each triple of atoms is repeated N
                            // times in the extended system through translational symmetry. The
                            // weight for the angles is thus divided by N so that the
                            // multiplication from symmetry is countered. This makes the final
                            // spectrum invariant to the selected supercell size and shape
                            // after normalization. The number of repetitions N is given by how
                            // many unique cell indices (the index of the repeated cell with
                            // respect to the original cell at index [0, 0, 0]) are present for
                            // the atoms in the triple.
                            if (!this->isLocal) {
                                vector<int> i_copy = this->cellIndices[i];
                                vector<int> j_copy = this->cellIndices[j];
                                vector<int> k_copy = this->cellIndices[k];

                                bool ij_equal = i_copy == j_copy;
                                bool ik_equal = i_copy == k_copy;
                                bool jk_equal = j_copy == k_copy;
                                int equal_sum = (int)ij_equal + (int)ik_equal + (int)jk_equal;

                                if (equal_sum == 1) {
                                    weight /= 2;
                                } else if (equal_sum == 0) {
                                    weight /= 3;
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
                            if (k_index < i_index) {
                                int temp = k_index;
                                k_index = i_index;
                                i_index = temp;
                            }

                            // Form the key as string to enable passing it through cython
                            stringstream ss;
                            ss << i_index;
                            ss << ",";
                            ss << j_index;
                            ss << ",";
                            ss << k_index;
                            string stringKey = ss.str();

                            // Sum gaussian into output
                            try {
                                vector<double> &old = k3Map.at(stringKey);
                                transform(old.begin(), old.end(), gauss.begin(), old.begin(), plus<double>());
                            } catch(const out_of_range& oor) {
                                k3Map[stringKey] = gauss;
                            }
                        }
                    }
                }
            }
        }
    }
    return k3Map;
}

inline vector<double> MBTR::gaussian(const double &center, const double &sigma, const double &weight, const double &min, const double &max, const int &n) {

    // Calculate CDF
    double dx = (max-min)/(n-1);
    vector<double> cdf(n+1);
    double sigmasqrt2 = sigma*sqrt(2.0);
    double x = min-dx/2;
    for (auto &it : cdf) {
        it = weight*1.0/2.0*(1.0 + erf((x-center)/sigmasqrt2));
        x += dx;
    }

    // Calculate PDF
    vector<double> pdf(n);
    int i = 0;
    for (auto &it : pdf) {
        it = (cdf[i+1]-cdf[i])/dx;
        ++i;
    }

    return pdf;
}

vector<index1d> MBTR::getk1Indices()
{
    // Use cached value if possible
    if (!this->k1IndicesInitialized) {

        int nAtoms = this->atomicNumbers.size();

        // Initialize the map containing the mappings
        vector<index1d> indexList;

        for (int i=0; i < nAtoms; ++i) {
            // Only consider atoms within the original cell
            if (i < this->interactionLimit) {
                index1d key(i);
                indexList.push_back(key);
            }
        }
        this->k1Indices = indexList;
        this->k1IndicesInitialized = true;
    }
    return this->k1Indices;
}

vector<index2d> MBTR::getk2Indices(const vector<vector<int> > &neighbours)
{
    // Use cached value if possible
    if (!this->k2IndicesInitialized) {

        int nAtoms = this->atomicNumbers.size();

        // Initialize the map containing the mappings
        vector<index2d> indexList;

        // We have to loop over all atoms in the system
        for (int i=0; i < nAtoms; ++i) {

            // For each atom we loop only over the neighbours
            const vector<int> &i_neighbours = neighbours[i];
            for (const int &j : i_neighbours) {
                if (j > i) {

                    // Only consider triplets that have one atom in the original
                    // cell
                    if (i < this->interactionLimit || j < this->interactionLimit) {
                        index2d key(i, j);
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

vector<index3d> MBTR::getk3Indices(const vector<vector<int> > &neighbours)
{
    // Use cached value if possible
    if (!this->k3IndicesInitialized) {

        int nAtoms = this->atomicNumbers.size();

        // Initialize the map containing the mappings
        vector<index3d> indexList;

        for (int i=0; i < nAtoms; ++i) {

            // For each atom we loop only over the atoms triplest that are
            // within the neighbourhood
            const vector<int> &i_neighbours = neighbours[i];
            for (const int &j : i_neighbours) {
                for (const int &k : i_neighbours) {
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
                                index3d key(i, j, k);
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

inline float MBTR::k1GeomAtomicNumber(int &i)
{
    int atomicNumber = this->atomicNumbers[i];
    return atomicNumber;
}

inline float MBTR::k1WeightUnity(int &i)
{
    return 1;
}

inline float MBTR::k2GeomInverseDistance(int &i, const int &j, const vector<vector<double> > &distances)
{
    float dist = k2GeomDistance(i, j, distances);
    float invDist = 1/dist;
    return invDist;
}

inline float MBTR::k2GeomDistance(int &i, const int &j, const vector<vector<double> > &distances)
{
    float dist = distances[i][j];
    return dist;
}

inline float MBTR::k2WeightUnity(int &i, const int &j, const vector<vector<double> > &distances)
{
    return 1;
}

inline float MBTR::k2WeightExponential(int &i, const int &j, const vector<vector<double> > &distances, float &scale, float &cutoff)
{
    float dist = distances[i][j];
    float expValue = exp(-scale*dist);
    if (expValue < cutoff) {
        expValue = 0;
    }
    return expValue;
}

inline float MBTR::k3GeomCosine(int &i, const int &j, const int &k, const vector<vector<double> > &distances)
{
    float r_ji = distances[j][i];
    float r_ik = distances[i][k];
    float r_jk = distances[j][k];
    float r_ji_square = r_ji*r_ji;
    float r_ik_square = r_ik*r_ik;
    float r_jk_square = r_jk*r_jk;
    float cosine = 0.5/(r_jk*r_ji) * (r_ji_square+r_jk_square-r_ik_square);

    // Due to numerical reasons the cosine might be slighlty under -1 or
    // above 1 degrees. E.g. acos is not defined then so we clip the values
    // to prevent NaN:s
    cosine = max(-1.0f, min(cosine, 1.0f));

    return cosine;
}

inline float MBTR::k3GeomAngle(int &i, const int &j, const int &k, const vector<vector<double> > &distances)
{
    float cosine = this->k3GeomCosine(i, j, k, distances);
    float angle = acos(cosine)*180.0/PI;

    return angle;
}

inline float MBTR::k3WeightExponential(int &i, const int &j, const int &k, const vector<vector<double> > &distances, float &scale, float &cutoff)
{
    float dist1 = distances[i][j];
    float dist2 = distances[j][k];
    float dist3 = distances[k][i];
    float distTotal = dist1 + dist2 + dist3;
    float expValue = exp(-scale*distTotal);
    if (expValue < cutoff) {
        expValue = 0;
    }

    return expValue;
}

inline float MBTR::k3WeightUnity(int &i, const int &j, const int &k, const vector<vector<double> > &distances)
{
    return 1;
}


map<index1d, float> MBTR::k1GeomAtomicNumber(const vector<index1d> &indexList)
{
    map<index1d,float> valueMap;
    for (const index1d& index : indexList) {
        int i = get<0>(index);
        int atomicNumber = this->atomicNumbers[i];
        valueMap[index] = atomicNumber;
    }

    return valueMap;
}

map<index2d, float> MBTR::k2GeomInverseDistance(const vector<index2d> &indexList, const vector<vector<float> > &distances)
{
    map<index2d, float> valueMap = this->k2GeomDistance(indexList, distances);
    for(auto& item : valueMap) {
        item.second = 1/item.second;
    }
    return valueMap;
}

map<index2d, float> MBTR::k2GeomDistance(const vector<index2d> &indexList, const vector<vector<float> > &distances)
{
    map<index2d,float> valueMap;
    for (const index2d& index : indexList) {
        int i = get<0>(index);
        int j = get<1>(index);

        float dist = distances[i][j];
        valueMap[index] = dist;
    }

    return valueMap;
}

map<index3d, float> MBTR::k3GeomCosine(const vector<index3d> &indexList, const vector<vector<float> > &distances)
{
    map<index3d,float> valueMap;
    for (const index3d& index : indexList) {
        int i = get<0>(index);
        int j = get<1>(index);
        int k = get<2>(index);
        float r_ji = distances[j][i];
        float r_ik = distances[i][k];
        float r_jk = distances[j][k];
        float r_ji_square = r_ji*r_ji;
        float r_ik_square = r_ik*r_ik;
        float r_jk_square = r_jk*r_jk;
        float cosine = 0.5/(r_jk*r_ji) * (r_ji_square+r_jk_square-r_ik_square);

        // Due to numerical reasons the cosine might be slighlty under -1 or
        // above 1 degrees. E.g. acos is not defined then so we clip the values
        // to prevent NaN:s
        cosine = max(-1.0f, min(cosine, 1.0f));

        valueMap[index] = cosine;
    }

    return valueMap;
}

map<index3d, float> MBTR::k3GeomAngle(const vector<index3d> &indexList, const vector<vector<float> > &distances)
{
    map<index3d, float> valueMap = this->k3GeomCosine(indexList, distances);
    for(auto& item : valueMap) {
        item.second = acos(item.second)*180.0/PI;
    }
    return valueMap;
}

map<index1d, float> MBTR::k1WeightUnity(const vector<index1d> &indexList)
{
    map<index1d, float> valueMap;

    for (const index1d& index : indexList) {
        valueMap[index] = 1;
    }

    return valueMap;
}

map<index2d, float> MBTR::k2WeightUnity(const vector<index2d> &indexList)
{
    map<index2d, float> valueMap;

    for (const index2d& index : indexList) {
        valueMap[index] = 1;
    }

    return valueMap;
}

map<index3d, float> MBTR::k3WeightUnity(const vector<index3d> &indexList)
{
    map<index3d, float> valueMap;

    for (const index3d& index : indexList) {
        valueMap[index] = 1;
    }

    return valueMap;
}

map<index2d, float> MBTR::k2WeightExponential(const vector<index2d> &indexList, float scale, float cutoff, const vector<vector<float> > &distances)
{
    map<index2d, float> valueMap;

    for (const index2d& index : indexList) {
        int i = get<0>(index);
        int j = get<1>(index);

        float dist = distances[i][j];
        float expValue = exp(-scale*dist);
        if (expValue >= cutoff) {
            valueMap[index] = expValue;
        }
    }

    return valueMap;
}

map<index3d, float> MBTR::k3WeightExponential(const vector<index3d> &indexList, float scale, float cutoff, const vector<vector<float> > &distances)
{
    map<index3d, float> valueMap;

    for (const index3d& index : indexList) {
        int i = get<0>(index);
        int j = get<1>(index);
        int k = get<2>(index);

        float dist1 = distances[i][j];
        float dist2 = distances[j][k];
        float dist3 = distances[k][i];
        float distTotal = dist1 + dist2 + dist3;
        float expValue = exp(-scale*distTotal);
        if (expValue >= cutoff) {
            valueMap[index] = expValue;
        }
    }

    return valueMap;
}

pair<map<index1d, vector<float> >, map<index1d,vector<float> > > MBTR::getK1GeomsAndWeights(string geomFunc, string weightFunc, map<string, float> parameters)
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
            int i = get<0>(index);

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
            index1d indexKey (i_index);

            // Save the values
            geomMap[indexKey].push_back(geomValue);
            weightMap[indexKey].push_back(weightValue);
        }

        this->k1Map = make_pair(geomMap, weightMap);
        this->k1MapInitialized = true;
    }
    return this->k1Map;
}

pair<map<index2d, vector<float> >, map<index2d,vector<float> > > MBTR::getK2GeomsAndWeights(const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters)
{
    // Use cached value if possible
    if (!this->k2IndicesInitialized) {

        vector<index2d> indexList = this->getk2Indices(neighbours);

        // Initialize the maps
        map<index2d, vector<float> > geomMap;
        map<index2d, vector<float> > weightMap;

        // Calculate all geometry values
        map<index2d, float> geomValues;
        if (geomFunc == "inverse_distance") {
            geomValues = this->k2GeomInverseDistance(indexList, distances);
        } else if (geomFunc == "distance") {
            geomValues = this->k2GeomDistance(indexList, distances);
        } else {
            throw invalid_argument("Invalid geometry function.");
        }

        // Calculate all weighting values
        map<index2d, float> weightValues;
        if (weightFunc == "exponential" || weightFunc == "exp") {
            float scale = parameters["scale"];
            float cutoff = parameters["cutoff"];
            weightValues = this->k2WeightExponential(indexList, scale, cutoff, distances);
        } else if (weightFunc == "unity") {
            weightValues = this->k2WeightUnity(indexList);
        } else {
            throw invalid_argument("Invalid weighting function.");
        }

        // Save the geometry and distances to the maps
        for (index2d& index : indexList) {
            int i = get<0>(index);
            int j = get<1>(index);

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
                vector<int> i_copy = this->cellIndices[i];
                vector<int> j_copy = this->cellIndices[j];

                if (i_copy != j_copy) {
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
                indexKey = index2d(j_index, i_index);
            } else {
                indexKey = index2d(i_index, j_index);
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

pair<map<index3d, vector<float> >, map<index3d,vector<float> > > MBTR::getK3GeomsAndWeights(const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters)
{
    // Use cached value if possible
    if (!this->k3IndicesInitialized) {

        vector<index3d> indexList = this->getk3Indices(neighbours);

        // Initialize the maps
        map<index3d, vector<float> > geomMap;
        map<index3d, vector<float> > weightMap;

        // Calculate all geometry values
        map<index3d, float> geomValues;
        if (geomFunc == "cosine") {
            geomValues = this->k3GeomCosine(indexList, distances);
        } else if (geomFunc == "angle") {
            geomValues = this->k3GeomAngle(indexList, distances);
        } else {
            throw invalid_argument("Invalid geometry function.");
        }

        // Calculate all weighting values
        map<index3d, float> weightValues;
        if (weightFunc == "exponential" || weightFunc == "exp") {
            float scale = parameters["scale"];
            float cutoff = parameters["cutoff"];
            weightValues = this->k3WeightExponential(indexList, scale, cutoff, distances);
        } else if (weightFunc == "unity") {
            weightValues = this->k3WeightUnity(indexList);
        } else {
            throw invalid_argument("Invalid weighting function.");
        }

        // Save the geometry and distances to the maps
        for (index3d& index : indexList) {
            int i = get<0>(index);
            int j = get<1>(index);
            int k = get<2>(index);

            // By default C++ map will return a default constructed float when
            // if the key is not found with the []-operator. Instead we
            // explicitly ask if the key is present.
            if ( weightValues.find(index) == weightValues.end() ) {
                continue;
            }

            float geomValue = geomValues[index];
            float weightValue = weightValues[index];

            // The contributions are weighted by their multiplicity arising from
            // the translational symmetry. Each triple of atoms is repeated N
            // times in the extended system through translational symmetry. The
            // weight for the angles is thus divided by N so that the
            // multiplication from symmetry is countered. This makes the final
            // spectrum invariant to the selected supercell size and shape
            // after normalization. The number of repetitions N is given by how
            // many unique cell indices (the index of the repeated cell with
            // respect to the original cell at index [0, 0, 0]) are present for
            // the atoms in the triple.
            if (!this->isLocal) {
                vector<int> i_copy = this->cellIndices[i];
                vector<int> j_copy = this->cellIndices[j];
                vector<int> k_copy = this->cellIndices[k];

                bool ij_equal = i_copy == j_copy;
                bool ik_equal = i_copy == k_copy;
                bool jk_equal = j_copy == k_copy;
                int equal_sum = (int)ij_equal + (int)ik_equal + (int)jk_equal;

                if (equal_sum == 1) {
                    weightValue /= 2;
                } else if (equal_sum == 0) {
                    weightValue /= 3;
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
                indexKey = index3d(k_index, j_index, i_index);
            } else {
                indexKey = index3d(i_index, j_index, k_index);
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

pair<map<string,vector<float> >, map<string,vector<float> > > MBTR::getK1GeomsAndWeightsCython(string geomFunc, string weightFunc, map<string, float> parameters)
{
    pair<map<index1d,vector<float> >, map<index1d,vector<float> > > cMap = this->getK1GeomsAndWeights(geomFunc, weightFunc, parameters);
    map<index1d, vector<float> > geomValues = cMap.first;
    map<index1d, vector<float> > weightValues = cMap.second;

    map<string, vector<float> > cythonGeom;
    map<string, vector<float> > cythonWeight;

    for (auto const& x : geomValues) {
        stringstream ss;
        ss << get<0>(x.first);
        string stringKey = ss.str();
        cythonGeom[stringKey] = x.second;
        cythonWeight[stringKey] = weightValues[x.first];
    }
    return make_pair(cythonGeom, cythonWeight);
}

pair<map<string,vector<float> >, map<string,vector<float> > > MBTR::getK2GeomsAndWeightsCython(const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters)
{
    pair<map<index2d,vector<float> >, map<index2d,vector<float> > > cMap = this->getK2GeomsAndWeights(distances, neighbours, geomFunc, weightFunc, parameters);
    map<index2d, vector<float> > geomValues = cMap.first;
    map<index2d, vector<float> > weightValues = cMap.second;

    map<string, vector<float> > cythonGeom;
    map<string, vector<float> > cythonWeight;

    for (auto const& x : geomValues) {
        stringstream ss;
        ss << get<0>(x.first);
        ss << ",";
        ss << get<1>(x.first);
        string stringKey = ss.str();
        cythonGeom[stringKey] = x.second;
        cythonWeight[stringKey] = weightValues[x.first];
    }
    return make_pair(cythonGeom, cythonWeight);
}

pair<map<string,vector<float> >, map<string,vector<float> > > MBTR::getK3GeomsAndWeightsCython(const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, string geomFunc, string weightFunc, map<string, float> parameters)
{
    pair<map<index3d,vector<float> >, map<index3d,vector<float> > > cMap = this->getK3GeomsAndWeights(distances, neighbours, geomFunc, weightFunc, parameters);
    map<index3d, vector<float> > geomValues = cMap.first;
    map<index3d, vector<float> > weightValues = cMap.second;

    map<string, vector<float> > cythonGeom;
    map<string, vector<float> > cythonWeight;

    for (auto const& x : geomValues) {
        stringstream ss;
        ss << get<0>(x.first);
        ss << ",";
        ss << get<1>(x.first);
        ss << ",";
        ss << get<2>(x.first);
        string stringKey = ss.str();
        cythonGeom[stringKey] = x.second;
        cythonWeight[stringKey] = weightValues[x.first];
    }
    return make_pair(cythonGeom, cythonWeight);
}
