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

vector<vector<float> > ACSF::create(vector<vector<float> > &positions, vector<int> &atomicNumbers, vector<vector<float> > &distances, vector<int> &indices)
{
    // Allocate memory
    int nIndices = indices.size();
    vector<vector<float> > output(nIndices, vector<float>((1+nG2+nG3)*nTypes+(nG4+nG5)*nTypePairs, 0));

    // Calculate the symmetry function values for every specified atom
    int nAtoms = atomicNumbers.size();
    int index = 0;
    for (int &i : indices) {
        for (int j=0; j < nAtoms; ++j) {
            if (i == j) {
                continue;
            }
            computeBond(output[index], atomicNumbers, distances, i, j);
            for (int k=0; k < j; ++k) {
                if (k == i) {
                    continue;
                }
                computeAngle(output[index], atomicNumbers, distances, i, j, k);
            }
        }
        ++index;
    }

    return output;
}


/*! \brief Computes the value of the cutoff fuction at a specific distance.
 * */
inline float ACSF::computeCutoff(float Rij) {
	return (Rij<rCut)? 0.5*(cos(Rij*PI/rCut)+1) : 0;
}

/*! \brief Compute the G1, G2 and G3 terms for atom ai with bi
 * */
void ACSF::computeBond(vector<float> &output, vector<int> &atomicNumbers, vector<vector<float> > &distances, int ai, int bi) {

	// index of type of B
	int bti = atomicNumberToIndexMap[atomicNumbers[bi]];

	// fetch distance from the matrix
	float Rij = distances[ai][bi];
	if (Rij >= rCut) {
	    return;
    }

	// Determine the location for this pair of species
	int offset = bti * (1+nG2+nG3);  // Skip G1, G2, G3 types that are not the ones of atom bi

	float val;
	float eta;
	float Rs;

	// Compute G1 - first function is just the cutoffs
	float fc = computeCutoff(Rij);
	output[offset] += fc;
	offset += 1;

	// Compute G2 - gaussian types
	for (auto params : g2Params) {
        eta = params[0];
        Rs = params[1];
        output[offset] += exp(-eta * (Rij - Rs)*(Rij - Rs)) * fc;
        offset++;
	}

	// Compute G3 - cosine type
	for (auto param : g3Params) {
        val = cos(Rij*param)*fc;
        output[offset] += val;
        offset++;
    }
}

/*! \brief Compute the G4 and G5 terms for triplet i, j, k
 * */
void ACSF::computeAngle(vector<float> &output, vector<int> &atomicNumbers, vector<vector<float> > &distances, int i, int j, int k) {

	// index of type of B
	int typj = atomicNumberToIndexMap[atomicNumbers[j]];
	int typk = atomicNumberToIndexMap[atomicNumbers[k]];

	// fetch distance from matrix
	float Rij = distances[i][j];
	if (Rij >= rCut) {
	    return;
    }

	float Rik = distances[i][k];
	if (Rik >= rCut) {
	    return;
    }

	float Rjk = distances[k][j];
	if (Rjk >= rCut) {
	    return;
    }

	// Determine the location for this triplet of species
    int offset = 0;
	int its;
	if (typj >= typk) {
	    its = (typj*(typj+1))/2 + typk;
    } else  {
        its = (typk*(typk+1))/2 + typj;
    }
	offset += nTypes * (1+nG2+nG3);         // Skip this atoms G1 G2 and G3
	offset += its * (nG4+nG5);              // skip G4 and G5 types that are not the ones of atom bi

    float cutoff_ij = computeCutoff(Rij);
    float cutoff_ik = computeCutoff(Rik);
    float cutoff_jk = computeCutoff(Rjk);
	float fc4 = cutoff_ij*cutoff_ik*cutoff_jk;
	float fc5 = cutoff_ij*cutoff_ik;

	float costheta = 0.5/(Rij*Rik);
	Rij *= Rij; //square all distances!
	Rik *= Rik;
	Rjk *= Rjk;
	costheta = costheta * (Rij+Rik-Rjk);

	float eta, gauss, zeta, lambda;

	// Compute G4
	for (auto params : g4Params) {
		eta = params[0];
		zeta = params[1];
		lambda = params[2];
		gauss = exp(-eta*(Rij+Rik+Rjk)) * fc4;
		output[offset] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
		offset++;
	}

	// Compute G5
	for (auto params : g5Params) {
		eta = params[0];
		zeta = params[1];
		lambda = params[2];
		gauss = exp(-eta*(Rij+Rik)) * fc5;
		output[offset] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
		offset++;
	}
}
