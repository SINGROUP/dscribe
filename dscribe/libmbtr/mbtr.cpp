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

MBTR::MBTR(map<int,int> atomicNumberToIndexMap, int interactionLimit, vector<vector<int>> cellIndices)
    : atomicNumberToIndexMap(atomicNumberToIndexMap)
    , interactionLimit(interactionLimit)
    , cellIndices(cellIndices)
{
}

map<string, vector<float>> MBTR::getK1(const vector<int> &Z, const string &geomFunc, const string &weightFunc, const map<string, float> &parameters, float min, float max, float sigma, int n)
{
    map<string, vector<float>> k1Map;
    int nAtoms = Z.size();
    float dx = (max-min)/(n-1);
    float sigmasqrt2 = sigma*sqrt(2.0);
    float start = min-dx/2;

    for (int i=0; i < nAtoms; ++i) {
        // Only consider atoms within the original cell
        if (i < this->interactionLimit) {

            // Calculate geometry value
            float geom;
            if (geomFunc == "atomic_number") {
                geom = k1GeomAtomicNumber(i, Z);
            } else {
                throw invalid_argument("Invalid geometry function.");
            }

            // Calculate weight value
            float weight;
            if (weightFunc == "unity") {
                weight = k1WeightUnity(i);
            } else {
                throw invalid_argument("Invalid weighting function.");
            }

            // Calculate gaussian
            vector<float> gauss = gaussian(geom, weight, start, dx, sigmasqrt2, n);

            // Get the index of the present elements in the final vector
            int i_elem = Z[i];
            int i_index = this->atomicNumberToIndexMap.at(i_elem);

            // Form the key as string to enable passing it through cython
            string stringKey = to_string(i_index);

            // Sum gaussian into output
            auto it = k1Map.find(stringKey);
            if ( it == k1Map.end() ) {
                k1Map[stringKey] = gauss;
            } else {
                vector<float> &old = it->second;
                transform(old.begin(), old.end(), gauss.begin(), old.begin(), plus<float>());
            }
        }
    }
    return k1Map;
}

map<string, vector<float>> MBTR::getK2(const vector<int> &Z, const vector<vector<float>> &distances, const vector<vector<int>> &neighbours, const string &geomFunc, const string &weightFunc, const map<string, float> &parameters, float min, float max, float sigma, int n)
{
    // Initialize some variables outside the loop
    map<string, vector<float> > k2Map;
    int nAtoms = Z.size();
    float dx = (max-min)/(n-1);
    float sigmasqrt2 = sigma*sqrt(2.0);
    float start = min-dx/2;

    // We have to loop over all atoms in the system
    for (int i=0; i < nAtoms; ++i) {

        // For each atom we loop only over the neighbours
        const vector<int> &i_neighbours = neighbours[i];
        for (const int &j : i_neighbours) {
            if (j > i) {

                // Only consider pairs that have one atom in the original
                // cell
                if (i < this->interactionLimit || j < this->interactionLimit) {

                    // Calculate geometry value
                    float geom;
                    if (geomFunc == "inverse_distance") {
                        geom = k2GeomInverseDistance(i, j, distances);
                    } else if (geomFunc == "distance") {
                        geom = k2GeomDistance(i, j, distances);
                    } else {
                        throw invalid_argument("Invalid geometry function.");
                    }

                    // Calculate weight value
                    float weight;
                    if (weightFunc == "exponential" || weightFunc == "exp") {
                        float scale = parameters.at("scale");
                        float cutoff = parameters.at("cutoff");
                        weight = k2WeightExponential(i, j, distances, scale);
                        if (weight < cutoff) {
                            continue;
                        }
                    } else if (weightFunc == "unity") {
                        weight = k2WeightUnity(i, j, distances);
                    } else {
                        throw invalid_argument("Invalid weighting function.");
                    }

                    // Find position in output
                    // When the pair of atoms are in different copies of the cell, the
                    // weight is halved. This is done in order to avoid double counting
                    // the same distance in the opposite direction. This correction
                    // makes periodic cells with different translations equal and also
                    // supercells equal to the primitive cell within a constant that is
                    // given by the number of repetitions of the primitive cell in the
                    // supercell.
                    vector<int> i_copy = this->cellIndices[i];
                    vector<int> j_copy = this->cellIndices[j];
                    if (i_copy != j_copy) {
                        weight /= 2;
                    }

                    // Calculate gaussian
                    vector<float> gauss = gaussian(geom, weight, start, dx, sigmasqrt2, n);

                    // Get the index of the present elements in the final vector
                    int i_elem = Z[i];
                    int j_elem = Z[j];
                    int i_index = this->atomicNumberToIndexMap.at(i_elem);
                    int j_index = this->atomicNumberToIndexMap.at(j_elem);

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
                    auto it = k2Map.find(stringKey);
                    if ( it == k2Map.end() ) {
                        k2Map[stringKey] = gauss;
                    } else {
                        vector<float> &old = it->second;
                        transform(old.begin(), old.end(), gauss.begin(), old.begin(), plus<float>());
                    }
                }
            }
        }
    }

    return k2Map;
}

map<string, vector<float> > MBTR::getK3(const vector<int> &Z, const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, const string &geomFunc, const string &weightFunc, const map<string, float> &parameters, float min, float max, float sigma, int n)
{
    map<string, vector<float> > k3Map;
    int nAtoms = Z.size();
    float dx = (max-min)/(n-1);
    float sigmasqrt2 = sigma*sqrt(2.0);
    float start = min-dx/2;

    for (int i=0; i < nAtoms; ++i) {

        // For each atom we loop only over the atoms triplets that are
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

                            // Calculate geometry value
                            float geom;
                            if (geomFunc == "cosine") {
                                geom = k3GeomCosine(i, j, k, distances);
                            } else if (geomFunc == "angle") {
                                geom = k3GeomAngle(i, j, k, distances);
                            } else {
                                throw invalid_argument("Invalid geometry function.");
                            }

                            // Calculate weight value
                            float weight;
                            if (weightFunc == "exponential" || weightFunc == "exp") {
                                float scale = parameters.at("scale");
                                float cutoff = parameters.at("cutoff");
                                weight = k3WeightExponential(i, j, k, distances, scale);
                                if (weight < cutoff) {
                                    continue;
                                }
                            } else if (weightFunc == "unity") {
                                weight = k3WeightUnity(i, j, k, distances);
                            } else {
                                throw invalid_argument("Invalid weighting function.");
                            }

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
                            vector<int> i_copy = this->cellIndices[i];
                            vector<int> j_copy = this->cellIndices[j];
                            vector<int> k_copy = this->cellIndices[k];
                            bool ij_diff = i_copy != j_copy;
                            bool ik_diff = i_copy != k_copy;
                            bool jk_diff = j_copy != k_copy;
                            int diff_sum = (int)ij_diff + (int)ik_diff + (int)jk_diff;
                            if (diff_sum > 1) {
                                weight /= diff_sum;
                            }

                            // Calculate gaussian
                            vector<float> gauss = gaussian(geom, weight, start, dx, sigmasqrt2, n);

                            // Get the index of the present elements in the final vector
                            int i_elem = Z[i];
                            int j_elem = Z[j];
                            int k_elem = Z[k];
                            int i_index = this->atomicNumberToIndexMap.at(i_elem);
                            int j_index = this->atomicNumberToIndexMap.at(j_elem);
                            int k_index = this->atomicNumberToIndexMap.at(k_elem);

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
                            auto it = k3Map.find(stringKey);
                            if ( it == k3Map.end() ) {
                                k3Map[stringKey] = gauss;
                            } else {
                                vector<float> &old = it->second;
                                transform(old.begin(), old.end(), gauss.begin(), old.begin(), plus<float>());
                            }
                        }
                    }
                }
            }
        }
    }
    return k3Map;
}

inline vector<float> MBTR::gaussian(float center, float weight, float start, float dx, float sigmasqrt2, int n) {

    // We first calculate the cumulative distibution function for a normal
    // distribution.
    vector<float> cdf(n+1);
    float x = start;
    for (auto &it : cdf) {
        it = weight*1.0/2.0*(1.0 + erf((x-center)/sigmasqrt2));
        x += dx;
    }

    // The normal distribution is calculated as a derivative of the cumulative
    // distribution, as with coarse discretization this methods preserves the
    // norm better.
    vector<float> pdf(n);
    int i = 0;
    for (auto &it : pdf) {
        it = (cdf[i+1]-cdf[i])/dx;
        ++i;
    }

    return pdf;
}

inline float MBTR::k1GeomAtomicNumber(const int &i, const vector<int> &Z)
{
    int atomicNumber = Z[i];
    return atomicNumber;
}

inline float MBTR::k1WeightUnity(const int &i)
{
    return 1;
}

inline float MBTR::k2GeomInverseDistance(const int &i, const int &j, const vector<vector<float> > &distances)
{
    float dist = k2GeomDistance(i, j, distances);
    float invDist = 1/dist;
    return invDist;
}

inline float MBTR::k2GeomDistance(const int &i, const int &j, const vector<vector<float> > &distances)
{
    float dist = distances[i][j];
    return dist;
}

inline float MBTR::k2WeightUnity(const int &i, const int &j, const vector<vector<float> > &distances)
{
    return 1;
}

inline float MBTR::k2WeightExponential(const int &i, const int &j, const vector<vector<float> > &distances, float scale)
{
    float dist = distances[i][j];
    float expValue = exp(-scale*dist);
    return expValue;
}

inline float MBTR::k3GeomCosine(const int &i, const int &j, const int &k, const vector<vector<float> > &distances)
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

inline float MBTR::k3GeomAngle(const int &i, const int &j, const int &k, const vector<vector<float> > &distances)
{
    float cosine = this->k3GeomCosine(i, j, k, distances);
    float angle = acos(cosine)*180.0/PI;

    return angle;
}

inline float MBTR::k3WeightExponential(const int &i, const int &j, const int &k, const vector<vector<float> > &distances, float scale)
{
    float dist1 = distances[i][j];
    float dist2 = distances[j][k];
    float dist3 = distances[k][i];
    float distTotal = dist1 + dist2 + dist3;
    float expValue = exp(-scale*distTotal);

    return expValue;
}

inline float MBTR::k3WeightUnity(const int &i, const int &j, const int &k, const vector<vector<float> > &distances)
{
    return 1;
}

vector<map<string, vector<float>>> MBTR::getK2Local(const vector<int> &indices, const vector<int> &Z, const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, const string &geomFunc, const string &weightFunc, const map<string, float> &parameters, float min, float max, float sigma, int n)
{
    // Initialize some variables outside the loop
    int nPos = indices.size();
    vector<map<string, vector<float> > > k2Maps(nPos);
    float dx = (max-min)/(n-1);
    float sigmasqrt2 = sigma*sqrt(2.0);
    float start = min-dx/2;

    // We loop over the specified indices
    for (int i=0; i < nPos; ++i) {
        int iTrue = indices[i];
        map<string, vector<float> > k2Map;

        // For each atom we loop only over the neighbours
        const vector<int> &i_neighbours = neighbours[i];
        for (const int &j : i_neighbours) {

            // Self-distances are not considered
            if (iTrue == j) {
                continue;
            }

            // Calculate geometry value
            float geom;
            if (geomFunc == "inverse_distance") {
                geom = k2GeomInverseDistance(i, j, distances);
            } else if (geomFunc == "distance") {
                geom = k2GeomDistance(i, j, distances);
            } else {
                throw invalid_argument("Invalid geometry function.");
            }

            // Calculate weight value
            float weight;
            if (weightFunc == "exponential" || weightFunc == "exp") {
                float scale = parameters.at("scale");
                float cutoff = parameters.at("cutoff");
                weight = k2WeightExponential(i, j, distances, scale);
                if (weight < cutoff) {
                    continue;
                }
            } else if (weightFunc == "unity") {
                weight = k2WeightUnity(i, j, distances);
            } else {
                throw invalid_argument("Invalid weighting function.");
            }

            // Calculate gaussian
            vector<float> gauss = gaussian(geom, weight, start, dx, sigmasqrt2, n);

            // Get the index of the present elements in the final vector
            int i_elem = 0;
            int j_elem = Z[j];
            int i_index = this->atomicNumberToIndexMap.at(i_elem);
            int j_index = this->atomicNumberToIndexMap.at(j_elem);

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
            auto it = k2Map.find(stringKey);
            if ( it == k2Map.end() ) {
                k2Map[stringKey] = gauss;
            } else {
                vector<float> &old = it->second;
                transform(old.begin(), old.end(), gauss.begin(), old.begin(), plus<float>());
            }
        }
        k2Maps[i] = k2Map;
    }

    return k2Maps;
}

vector<map<string, vector<float>>> MBTR::getK3Local(const vector<int> &indices, const vector<int> &Z, const vector<vector<float>> &distances, const vector<vector<int>> &neighbours, const string &geomFunc, const string &weightFunc, const map<string, float> &parameters, float min, float max, float sigma, int n)
{
    // Initialize some variables outside the loop
    vector<map<string, vector<float>>> k3Maps(indices.size());
    float dx = (max-min)/(n-1);
    float sigmasqrt2 = sigma*sqrt(2.0);
    float start = min-dx/2;
    int iLoc = 0;

    // We loop over the specified indices
    for (const int &i : indices) {
        map<string, vector<float> > k3Map;

        // For each atom we loop only over the atoms triplets that are
        // within the neighbourhood
        const vector<int> &i_neighbours = neighbours[i];
        for (const int &j : i_neighbours) {
            for (const int &k : i_neighbours) {
                // Calculate angle for all index permutations from choosing
                // three out of nAtoms. The same atom cannot be present twice
                // in the permutation.
                if (j != i && k != j && k != i) {

                    // Calculate geometry value
                    float geom;
                    if (geomFunc == "cosine") {
                        geom = k3GeomCosine(i, j, k, distances);
                    } else if (geomFunc == "angle") {
                        geom = k3GeomAngle(i, j, k, distances);
                    } else {
                        throw invalid_argument("Invalid geometry function.");
                    }

                    // Calculate weight value
                    float weight;
                    if (weightFunc == "exponential" || weightFunc == "exp") {
                        float scale = parameters.at("scale");
                        float cutoff = parameters.at("cutoff");
                        weight = k3WeightExponential(i, j, k, distances, scale);
                        if (weight < cutoff) {
                            continue;
                        }
                    } else if (weightFunc == "unity") {
                        weight = k3WeightUnity(i, j, k, distances);
                    } else {
                        throw invalid_argument("Invalid weighting function.");
                    }

                    // Calculate gaussian
                    vector<float> gauss = gaussian(geom, weight, start, dx, sigmasqrt2, n);

                    // Get the index of the present elements in the final vector
                    int i_elem = 0;
                    int j_elem = Z[j];
                    int k_elem = Z[k];
                    int i_index = this->atomicNumberToIndexMap.at(i_elem);
                    int j_index = this->atomicNumberToIndexMap.at(j_elem);
                    int k_index = this->atomicNumberToIndexMap.at(k_elem);

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
                    auto it = k3Map.find(stringKey);
                    if ( it == k3Map.end() ) {
                        k3Map[stringKey] = gauss;
                    } else {
                        vector<float> &old = it->second;
                        transform(old.begin(), old.end(), gauss.begin(), old.begin(), plus<float>());
                    }

                    // Also include the angle where the local center is in
                    // the middle. Include it only once.
                    if (k > j) {
                        // Calculate geometry value
                        float geom;
                        if (geomFunc == "cosine") {
                            geom = k3GeomCosine(j, i, k, distances);
                        } else if (geomFunc == "angle") {
                            geom = k3GeomAngle(j, i, k, distances);
                        } else {
                            throw invalid_argument("Invalid geometry function.");
                        }

                        // Calculate weight value
                        float weight;
                        if (weightFunc == "exponential" || weightFunc == "exp") {
                            float scale = parameters.at("scale");
                            float cutoff = parameters.at("cutoff");
                            weight = k3WeightExponential(j, i, k, distances, scale);
                            if (weight < cutoff) {
                                continue;
                            }
                        } else if (weightFunc == "unity") {
                            weight = k3WeightUnity(j, i, k, distances);
                        } else {
                            throw invalid_argument("Invalid weighting function.");
                        }

                        // Calculate gaussian
                        vector<float> gauss = gaussian(geom, weight, start, dx, sigmasqrt2, n);

                        // Get the index of the present elements in the final vector
                        int i_elem = 0;
                        int j_elem = Z[j];
                        int k_elem = Z[k];
                        int i_index = this->atomicNumberToIndexMap.at(i_elem);
                        int j_index = this->atomicNumberToIndexMap.at(j_elem);
                        int k_index = this->atomicNumberToIndexMap.at(k_elem);

                        // Save information in the part where k_index >= j_index
                        if (k_index < j_index) {
                            int temp = k_index;
                            k_index = j_index;
                            j_index = temp;
                        }

                        // Form the key as string to enable passing it through cython
                        stringstream ss;
                        ss << j_index;
                        ss << ",";
                        ss << i_index;
                        ss << ",";
                        ss << k_index;
                        string stringKey = ss.str();

                        // Sum gaussian into output
                        auto it = k3Map.find(stringKey);
                        if ( it == k3Map.end() ) {
                            k3Map[stringKey] = gauss;
                        } else {
                            vector<float> &old = it->second;
                            transform(old.begin(), old.end(), gauss.begin(), old.begin(), plus<float>());
                        }
                    }
                }
            }
        }
        k3Maps[iLoc] = k3Map;
        ++iLoc;
    }
    return k3Maps;
}
