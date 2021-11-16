#include "acsf.h"
#include <tuple>
#include <map>
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

ACSF::ACSF(double rCut, vector<vector<double> > g2Params, vector<double> g3Params, vector<vector<double> > g4Params, vector<vector<double> > g5Params, vector<int> atomicNumbers)
{
    setRCut(rCut);
    setG2Params(g2Params);
    setG3Params(g3Params);
    setG4Params(g4Params);
    setG5Params(g5Params);
    setAtomicNumbers(atomicNumbers);
}

void ACSF::setRCut(double rCut)
{
    this->rCut = rCut;
}
double ACSF::getRCut()
{
return this->rCut;
}

void ACSF::setG2Params(vector<vector<double> > g2Params)
{
    this->g2Params = g2Params;
    nG2 = g2Params.size();
}

vector<vector<double> > ACSF::getG2Params()
{
return this->g2Params;
}


void ACSF::setG3Params(vector<double> g3Params)
{
    this->g3Params = g3Params;
    nG3 = g3Params.size();
}
vector<double> ACSF::getG3Params()
{
return this->g3Params;
}


void ACSF::setG4Params(vector<vector<double> > g4Params)
{
    this->g4Params = g4Params;
    nG4 = g4Params.size();
}
vector<vector<double> > ACSF::getG4Params()
{
return this->g4Params;
}


void ACSF::setG5Params(vector<vector<double> > g5Params)
{
    this->g5Params = g5Params;
    nG5 = g5Params.size();
}
vector<vector<double> > ACSF::getG5Params()
{
return this->g5Params;
}


void ACSF::setAtomicNumbers(vector<int> atomicNumbers)
{
    this->atomicNumbers = atomicNumbers;
    nTypes = atomicNumbers.size();
    nTypePairs = nTypes*(nTypes+1)/2;
    unordered_map<int, int> atomicNumberToIndexMap;
    int i = 0;
    for (int Z : atomicNumbers) {
        atomicNumberToIndexMap[Z] = i;
        ++i;
    }
    this->atomicNumberToIndexMap = atomicNumberToIndexMap;
}
vector<int> ACSF::getAtomicNumbers()
{
return this->atomicNumbers;
}


vector<vector<double> > ACSF::create(vector<vector<double> > &positions, vector<int> &atomicNumbers, const vector<vector<double> > &distances, const vector<vector<int> > &neighbours, vector<int> &indices)
{

    // Allocate memory
    int nIndices = indices.size();
    vector<vector<double> > output(nIndices, vector<double>((1+nG2+nG3)*nTypes+(nG4+nG5)*nTypePairs, 0));

    // Calculate the symmetry function values for every specified atom
    int index = 0;
    for (int &i : indices) {

        // Compute pairwise terms only for neighbors within cutoff
        const vector<int> &i_neighbours = neighbours[i];
        vector<double> &row = output[index];
        for (const int &j : i_neighbours) {
            if (i == j) {
                continue;
            }

            // Precompute some values
            double r_ij = distances[i][j];
            double fc_ij = computeCutoff(r_ij);
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
                for (const int &k : i_neighbours) {
                    if (k == i || k >= j) {
                        continue;
                    }

                    // Precompute some values that are used by both G4 and G5
                    double r_ik = distances[i][k];
                    double r_jk = distances[j][k];
                    double fc_ik = computeCutoff(r_ik);
                    double r_ij_square = r_ij*r_ij;
                    double r_ik_square = r_ik*r_ik;
                    double r_jk_square = r_jk*r_jk;
                    int index_k = atomicNumberToIndexMap[atomicNumbers[k]];
                    double costheta = 0.5/(r_ij*r_ik) * (r_ij_square+r_ik_square-r_jk_square);

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
inline double ACSF::computeCutoff(double r_ij) {
	return 0.5*(cos(r_ij*PI/rCut)+1);
}

inline void ACSF::computeG1(vector<double> &output, int &offset, double &fc_ij) {
    output[offset] += fc_ij;
    offset += 1;
}

inline void ACSF::computeG2(vector<double> &output, int &offset, double &r_ij, double &fc_ij) {

	// Compute G2 - gaussian types
    double eta;
    double Rs;
	for (auto params : g2Params) {
        eta = params[0];
        Rs = params[1];
        output[offset] += exp(-eta * (r_ij - Rs)*(r_ij - Rs)) * fc_ij;
        offset++;
	}
}

inline void ACSF::computeG3(vector<double> &output, int &offset, double &r_ij, double &fc_ij) {
	// Compute G3 - cosine type
	for (auto param : g3Params) {
        output[offset] += cos(r_ij*param)*fc_ij;
        offset++;
    }
}

inline void ACSF::computeG4(vector<double> &output, int &offset, double &costheta, double &r_jk, double &r_ij_square, double &r_ik_square, double &r_jk_square, double &fc_ij, double &fc_ik) {
	// Compute G4
    if (r_jk > rCut) {
        offset += g4Params.size();
        return;
    }
    double cutoff_jk = computeCutoff(r_jk);
	double fc4 = fc_ij*fc_ik*cutoff_jk;
	double eta;
	double zeta;
	double lambda;
	double gauss;
	for (auto params : g4Params) {
		eta = params[0];
		zeta = params[1];
		lambda = params[2];
		gauss = exp(-eta*(r_ij_square+r_ik_square+r_jk_square)) * fc4;
		output[offset] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
		offset++;
	}
}

inline void ACSF::computeG5(vector<double> &output, int &offset, double &costheta, double &r_ij_square, double &r_ik_square, double &fc_ij, double &fc_ik) {
	// Compute G5
	double eta;
	double zeta;
	double lambda;
	double gauss;
	double fc5 = fc_ij*fc_ik;
	for (auto params : g5Params) {
		eta = params[0];
		zeta = params[1];
		lambda = params[2];
		gauss = exp(-eta*(r_ij_square+r_ik_square)) * fc5;
		output[offset] += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
		offset++;
	}
}
