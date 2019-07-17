#include "acsf.h"
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
#include <iostream>
using namespace std;

ACSF::ACSF(float rCut, vector<vector<float> > g2Params, vector<float> g3Params, vector<vector<float> > g4Params, vector<vector<float> > g5Params, vector<int> atomicNumbers)
{
    setRCut(rCut);
    setG2Params(g2Params);
    setG3Params(g3Params);
    setG4Params(g4Params);
    setG5Params(g5Params);
    setAtomicNumbers(atomicNumbers);
}

void ACSF::setRCut(float rCut)
{
    this->rCut = rCut;
}

void ACSF::setG2Params(vector<vector<float> > g2Params)
{
    this->g2Params = g2Params;
    nG2 = g2Params.size();
}

void ACSF::setG3Params(vector<float> g3Params)
{
    this->g3Params = g3Params;
    nG3 = g3Params.size();
}

void ACSF::setG4Params(vector<vector<float> > g4Params)
{
    this->g4Params = g4Params;
    nG4 = g4Params.size();
}

void ACSF::setG5Params(vector<vector<float> > g5Params)
{
    this->g5Params = g5Params;
    nG5 = g5Params.size();
}

void ACSF::setAtomicNumbers(vector<int> atomicNumbers)
{
    this->atomicNumbers = atomicNumbers;
    nTypes = atomicNumbers.size();
    nTypePairs = nTypes*(nTypes+1)/2;
    map<int, int> atomicNumberToIndexMap;
    int i = 0;
    for (int Z : atomicNumbers) {
        atomicNumberToIndexMap[Z] = i;
        ++i;
    }
    this->atomicNumberToIndexMap = atomicNumberToIndexMap;
}

vector<vector<float> > ACSF::create(vector<vector<float> > &positions, vector<int> &atomicNumbers, vector<vector<float> > &distances, map<int, vector<int> > &neighbours, vector<int> &indices)
{
    // Allocate memory
    int nIndices = indices.size();
    vector<vector<float> > output(nIndices, vector<float>((1+nG2+nG3)*nTypes+(nG4+nG5)*nTypePairs, 0));

    // Calculate the symmetry function values for every specified atom
    int index = 0;
    for (int &i : indices) {

        // Compute pairwise terms only for neighbors within cutoff
        vector<int> &i_neighbours = neighbours[i];
        vector<float> &row = output[index];
        for (int &j : i_neighbours) {
            if (i == j) {
                continue;
            }

            // Precompute some values
            float r_ij = distances[i][j];
            if (r_ij > rCut) {
                continue;
            }

            float fc_ij = computeCutoff(r_ij);
            int index_j = atomicNumberToIndexMap[atomicNumbers[j]];
            int offset = index_j * (1+nG2+nG3);  // Skip G1, G2, G3 types that are not the ones of atom bi

            // Compute G1
            computeG1(row, offset, fc_ij);

            // Compute G2
            computeG2(row, offset, r_ij, fc_ij);

            // Compute G3
            computeG3(row, offset, r_ij, fc_ij);

            // Compute angle terms only when both neighbors are within cutoff
            if (g4Params.size() != 0 || g5Params.size() != 0) {
                for (int &k : i_neighbours) {
                    if (k == i || k >= j) {
                        continue;
                    }

                    // Precompute some values that are used by both G4 and G5
                    float r_ik = distances[i][k];
                    float r_jk = distances[j][k];
                    if (r_ik > rCut) {
                        continue;
                    }

                    float fc_ik = computeCutoff(r_ik);
                    float r_ij_square = r_ij*r_ij;
                    float r_ik_square = r_ik*r_ik;
                    float r_jk_square = r_jk*r_jk;
                    int index_k = atomicNumberToIndexMap[atomicNumbers[k]];
                    float costheta = 0.5/(r_ij*r_ik) * (r_ij_square+r_ik_square-r_jk_square);

                    // Determine the location for this triplet of species
                    int its;
                    if (index_j >= index_k) {
                        its = (index_j*(index_j+1))/2 + index_k;
                    } else  {
                        its = (index_k*(index_k+1))/2 + index_j;
                    }
                    offset = nTypes * (1+nG2+nG3);         // Skip this atoms G1 G2 and G3
                    offset += its * (nG4+nG5);              // skip G4 and G5 types that are not the ones of atom bi

                    // Compute G4
                    computeG4(row, offset, costheta, r_jk, r_ij_square, r_ik_square, r_jk_square, fc_ij, fc_ik);

                    // Compute G5
                    computeG5(row, offset, costheta, r_ij_square, r_ik_square, fc_ij, fc_ik);
                }
            }
        }
        ++index;
    }

    return output;
}


/*! \brief Computes the value of the cutoff fuction at a specific distance.
 * */
inline float ACSF::computeCutoff(float r_ij) {
	return 0.5*(cos(r_ij*PI/rCut)+1);
}

inline void ACSF::computeG1(vector<float> &output, int &offset, float &fc_ij) {
    output[offset] += fc_ij;
    offset += 1;
}

inline void ACSF::computeG2(vector<float> &output, int &offset, float &r_ij, float &fc_ij) {

	// Compute G2 - gaussian types
    float eta;
    float Rs;
	for (auto params : g2Params) {
        eta = params[0];
        Rs = params[1];
        output[offset] += exp(-eta * (r_ij - Rs)*(r_ij - Rs)) * fc_ij;
        offset++;
	}
}

inline void ACSF::computeG3(vector<float> &output, int &offset, float &r_ij, float &fc_ij) {
	// Compute G3 - cosine type
	for (auto param : g3Params) {
        output[offset] += cos(r_ij*param)*fc_ij;
        offset++;
    }
}

inline void ACSF::computeG4(vector<float> &output, int &offset, float &costheta, float &r_jk, float &r_ij_square, float &r_ik_square, float &r_jk_square, float &fc_ij, float &fc_ik) {
	// Compute G4
    if (r_jk > rCut) {
        offset += g4Params.size();
        return;
    }
    float cutoff_jk = computeCutoff(r_jk);
	float fc4 = fc_ij*fc_ik*cutoff_jk;
	float eta;
	float zeta;
	float lambda;
	float gauss;
	for (auto params : g4Params) {
		eta = params[0];
		zeta = params[1];
		lambda = params[2];
		gauss = exp(-eta*(r_ij_square+r_ik_square+r_jk_square)) * fc4;
		output[offset] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
		offset++;
	}
}

inline void ACSF::computeG5(vector<float> &output, int &offset, float &costheta, float &r_ij_square, float &r_ik_square, float &fc_ij, float &fc_ik) {
	// Compute G5
	float eta;
	float zeta;
	float lambda;
	float gauss;
	float fc5 = fc_ij*fc_ik;
	for (auto params : g5Params) {
		eta = params[0];
		zeta = params[1];
		lambda = params[2];
		gauss = exp(-eta*(r_ij_square+r_ik_square)) * fc5;
		output[offset] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
		offset++;
	}
}

/*! \brief Compute the G1, G2 and G3 terms for atom ai with bi
 * */
//void ACSF::computeBond(vector<float> &output, vector<int> &atomicNumbers, vector<vector<float> > &distances, int ai, int bi) {

	//// index of type of B
	//int bti = atomicNumberToIndexMap[atomicNumbers[bi]];

	//// fetch distance from the matrix
	//float Rij = distances[ai][bi];
	//if (Rij >= rCut) {
		//return;
    //}

	//// Determine the location for this pair of species
	//int offset = bti * (1+nG2+nG3);  // Skip G1, G2, G3 types that are not the ones of atom bi

	//float val;
	//float eta;
	//float Rs;

	//// Compute G1 - first function is just the cutoffs
	//float fc = computeCutoff(Rij);
	//output[offset] += fc;
	//offset += 1;

	//// Compute G2 - gaussian types
	//for (auto params : g2Params) {
        //eta = params[0];
        //Rs = params[1];
        //output[offset] += exp(-eta * (Rij - Rs)*(Rij - Rs)) * fc;
        //offset++;
	//}

	//// Compute G3 - cosine type
	//for (auto param : g3Params) {
        //val = cos(Rij*param)*fc;
        //output[offset] += val;
        //offset++;
    //}
//}

/*! \brief Compute the G4 and G5 terms for triplet i, j, k
 * */
//void ACSF::computeAngle(vector<float> &output, vector<int> &atomicNumbers, vector<vector<float> > &distances, int i, int j, int k) {

	//// index of type of B
	//int typj = atomicNumberToIndexMap[atomicNumbers[j]];
	//int typk = atomicNumberToIndexMap[atomicNumbers[k]];

	//// fetch distance from matrix
	//float Rij = distances[i][j];
	//if (Rij >= rCut) {
		//return;
    //}

	//float Rik = distances[i][k];
	//if (Rik >= rCut) {
		//return;
    //}

	//float Rjk = distances[j][k];
	//if (Rjk >= rCut) {
		//return;
    //}

	//// Determine the location for this triplet of species
    //int offset = 0;
	//int its;
	//if (typj >= typk) {
		//its = (typj*(typj+1))/2 + typk;
    //} else  {
        //its = (typk*(typk+1))/2 + typj;
    //}
	//offset += nTypes * (1+nG2+nG3);         // Skip this atoms G1 G2 and G3
	//offset += its * (nG4+nG5);              // skip G4 and G5 types that are not the ones of atom bi

    //float cutoff_ij = computeCutoff(Rij);
    //float cutoff_ik = computeCutoff(Rik);
    //float cutoff_jk = computeCutoff(Rjk);
	//float fc4 = cutoff_ij*cutoff_ik*cutoff_jk;
	//float fc5 = cutoff_ij*cutoff_ik;

	//float costheta = 0.5/(Rij*Rik);
	//Rij *= Rij; //square all distances!
	//Rik *= Rik;
	//Rjk *= Rjk;
	//costheta = costheta * (Rij+Rik-Rjk);

	//float eta, gauss, zeta, lambda;

	//// Compute G4
	//for (auto params : g4Params) {
		//eta = params[0];
		//zeta = params[1];
		//lambda = params[2];
		//gauss = exp(-eta*(Rij+Rik+Rjk)) * fc4;
		//output[offset] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
		//offset++;
	//}

	//// Compute G5
	//for (auto params : g5Params) {
		//eta = params[0];
		//zeta = params[1];
		//lambda = params[2];
		//gauss = exp(-eta*(Rij+Rik)) * fc5;
		//output[offset] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
		//offset++;
	//}
//}
