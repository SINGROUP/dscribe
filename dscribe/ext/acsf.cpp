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

namespace py = pybind11;
using namespace std;

ACSF::ACSF(double rCut, vector<vector<double> > g2Params, vector<double> g3Params, vector<vector<double> > g4Params, vector<vector<double> > g5Params, vector<int> atomicNumbers) {
    setRCut(rCut);
    setG2Params(g2Params);
    setG3Params(g3Params);
    setG4Params(g4Params);
    setG5Params(g5Params);
    setAtomicNumbers(atomicNumbers);
}

void ACSF::setRCut(double rCut) {
    this->rCut = rCut;
}
double ACSF::getRCut() {return this->rCut;}

void ACSF::setG2Params(vector<vector<double> > g2Params) {
    this->g2Params = g2Params;
    nG2 = g2Params.size();
}

vector<vector<double> > ACSF::getG2Params() {return this->g2Params;}

void ACSF::setG3Params(vector<double> g3Params) {
    this->g3Params = g3Params;
    nG3 = g3Params.size();
}
vector<double> ACSF::getG3Params() {return this->g3Params;}


void ACSF::setG4Params(vector<vector<double> > g4Params) {
    this->g4Params = g4Params;
    nG4 = g4Params.size();
}
vector<vector<double> > ACSF::getG4Params() {return this->g4Params;}


void ACSF::setG5Params(vector<vector<double> > g5Params) {
    this->g5Params = g5Params;
    nG5 = g5Params.size();
}
vector<vector<double> > ACSF::getG5Params() {return this->g5Params;}


void ACSF::setAtomicNumbers(vector<int> atomicNumbers) {
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
vector<int> ACSF::getAtomicNumbers() {return this->atomicNumbers;}


vector<vector<double>> ACSF::create(vector<vector<double>> &positions, vector<int> &atomicNumbers, const vector<vector<double> > &distances, const vector<vector<int>> &neighbours, vector<int> &indices) {

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


/*! \brief Computes the value of the cutoff fuction at a specific distance. */
inline double ACSF::computeCutoff(double r_ij) {
	return 0.5*(cos(r_ij*PI/rCut)+1);
}
inline void ACSF::computeG1(vector<double> &output, int &offset, double &fc_ij) {
    output[offset] += fc_ij;
    offset += 1;
}
inline void ACSF::computeG1_pyarray(py::detail::unchecked_mutable_reference<double,2> descriptor_mu, int &center, int &offset, double &fc_ij) {
    descriptor_mu(center, offset) += fc_ij;
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
inline void ACSF::computeG2_pyarray(py::detail::unchecked_mutable_reference<double,2> descriptor_mu, int &center, int &offset, double &r_ij, double &fc_ij) {

    // Compute G2 - gaussian types
    double eta;
    double Rs;
    for (auto params : g2Params) {
        eta = params[0];
        Rs = params[1];
        descriptor_mu(center, offset) += exp(-eta * (r_ij - Rs)*(r_ij - Rs)) * fc_ij;
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
inline void ACSF::computeG3_pyarray(py::detail::unchecked_mutable_reference<double,2> descriptor_mu, int &center, int &offset, double &r_ij, double &fc_ij) {
    // Compute G3 - cosine type
    for (auto param : g3Params) {
        descriptor_mu(center, offset) += cos(r_ij*param)*fc_ij;
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
inline void ACSF::computeG4_pyarray(py::detail::unchecked_mutable_reference<double,2> descriptor_mu, int &center, int &offset, double &costheta, double &r_jk, double &r_ij_square, double &r_ik_square, double &r_jk_square, double &fc_ij, double &fc_ik) {
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
        descriptor_mu(center, offset) += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
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
inline void ACSF::computeG5_pyarray(py::detail::unchecked_mutable_reference<double,2> descriptor_mu, int &center, int &offset, double &costheta, double &r_ij_square, double &r_ik_square, double &fc_ij, double &fc_ik) {
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
        descriptor_mu(center, offset) += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
        offset++;
    }
}


void ACSF::derivatives_analytical(
    py::array_t<double> derivatives,
    py::array_t<double> descriptor,
    py::array_t<int> atomic_numbers,
    py::array_t<double> atomic_positions,
    py::array_t<double> distances,
    const vector<vector<int>> &neighbours,
    py::array_t<int> desc_centers,
    py::array_t<int> grad_centers, // we want derivatives w.r.t. these atoms
    const bool return_descriptor
) {


    int n_desc_centers = desc_centers.shape(0);

    // TODO: the code!
    auto descriptor_mu = descriptor.mutable_unchecked<2>(); // [n_desc_centers, n_features]
    auto derivatives_mu = derivatives.mutable_unchecked<4>();
    

    auto atomic_numbers_u = atomic_numbers.unchecked<1>();
    auto atomic_positions_u = atomic_positions.unchecked<2>();

    auto distances_u    = distances.unchecked<2>();
    auto desc_centers_u = desc_centers.unchecked<1>();
    auto grad_centers_u = grad_centers.unchecked<1>();

    // d[i,j,k,l] = derivative of l-th ACSF of atom i, w.r.t. atom j coordinate k
    // c[i,l]     = l-th ACSF of atom i  [descriptor]

    for(int idxi=0; idxi<n_desc_centers; idxi++) {
        
        int i = desc_centers_u(idxi);
        const vector<int> &i_neighbours = neighbours[i];

        //printf("center[%i] = %i \n", idxi, i);

        // maybe here we can estimate which grad_centers affect the ACSFs of atom i

        for (const int &x : i_neighbours) {
            if(i == x) continue;

            //printf("\t with neighbour %i \n", x);

            // Precompute some values
            double r_ij = distances_u(i, x);
            double fc_ij = computeCutoff(r_ij);
            int index_j = atomicNumberToIndexMap[atomicNumbers[x]];
            int offset = index_j * (1+nG2+nG3);  // Skip G1, G2, G3 types that are not the ones of atom bi
            int o0 = offset;
            // offset is the feature offset!
            double e_ij[3];
            for(int c=0; c<3; c++)
                e_ij[c] = (atomic_positions_u(i,c) - atomic_positions_u(x,c)) / r_ij;
            
            if(return_descriptor){
                computeG1_pyarray(descriptor_mu, idxi, offset, fc_ij);
                computeG2_pyarray(descriptor_mu, idxi, offset, r_ij, fc_ij);
                computeG3_pyarray(descriptor_mu, idxi, offset, r_ij, fc_ij);
            }

            // G1
            // we have to look at the grad_centers_u
            // in there we might find i or x, in which case there is a contribution to the derivative

            // obviously this will contribute to d[i,i,*,offset?]
            // and other cntributions to d[i,j,*,offset?] when j == x
            // d[i,j,k,l] = derivative of l-th ACSF of atom i, w.r.t. atom j coordinate k
            double d_fc_ij = -(PI*sin((PI*r_ij)/rCut))/(2.0*rCut); // deriv of cutoff function w.r.t. ri

            // G1
            for(int idxj=0; idxj<grad_centers.shape(0); idxj++){
                int j = grad_centers_u(idxj);
                if(j != i && j != x) continue;
                if(j == i){
                    for(int c=0;c<3;c++)
                        derivatives_mu(idxi,idxj,c,o0) += e_ij[c]*d_fc_ij;
                    //printf("g1 grad j=i %i %i [center %i] %lf\n", i,x,j,e_ij[0]*d_fc_ij);
                }
                else if(j == x){
                    for(int c=0;c<3;c++)
                        derivatives_mu(idxi,idxj,c,o0) -= e_ij[c]*d_fc_ij;
                    //printf("g1 grad j=x %i %i [center %i] %lf\n", i,x,j,-e_ij[0]*d_fc_ij);
                }
            }
            o0++;

            // G2
            double eta, zeta, lambda, ef, der, tmp;
            double Rs;
            for (auto params : g2Params) {
                eta = params[0];
                Rs = params[1];
                tmp = (r_ij - Rs);
                ef = exp(-eta * tmp*tmp);
                der = ef*(fc_ij*(-eta*2*tmp) + d_fc_ij);

                for(int idxj=0; idxj<grad_centers.shape(0); idxj++){
                    int j = grad_centers_u(idxj);
                    if(j != i && j != x) continue;
                    if(j == i){
                        for(int c=0;c<3;c++)
                            derivatives_mu(idxi,idxj,c,o0) += e_ij[c]*der;
                    }
                    else if(j==x){
                        for(int c=0;c<3;c++)
                            derivatives_mu(idxi,idxj,c,o0) -= e_ij[c]*der;
                    }
                }
                o0++;
            }

            // G3
            for (auto param : g3Params) {

                der = cos(r_ij*param)*d_fc_ij - param*sin(r_ij*param)*fc_ij;

                for(int idxj=0; idxj<grad_centers.shape(0); idxj++){
                    int j = grad_centers_u(idxj);
                    if(j != i && j != x) continue;
                    if(j == i){
                        for(int c=0;c<3;c++)
                            derivatives_mu(idxi,idxj,c,o0) += e_ij[c]*der;
                    }
                    else if(j==x){
                        for(int c=0;c<3;c++)
                            derivatives_mu(idxi,idxj,c,o0) -= e_ij[c]*der;
                    }
                }
                o0++;
            }
            
            if (g4Params.size()==0 && g5Params.size()==0) continue;
            // code here => we have some 3-body terms
            
            for (const int &y : i_neighbours) {
                if (y == i || y >= x) continue;
                //printf("\t\t with neighbour %i [3body]\n", y);
                // Precompute some values that are used by both G4 and G5
                double r_ik = distances_u(i,y);
                double r_jk = distances_u(x,y);
                double fc_ik = computeCutoff(r_ik);
                double fc_jk = computeCutoff(r_jk);
                double r_ij_square = r_ij*r_ij;
                double r_ik_square = r_ik*r_ik;
                double r_jk_square = r_jk*r_jk;
                int index_k = atomicNumberToIndexMap[atomicNumbers[y]];
                double costheta = 0.5*(r_ij_square+r_ik_square-r_jk_square)/(r_ij*r_ik);
                double fc4 = fc_ij*fc_ik*fc_jk;
                double fc5 = fc_ij*fc_ik;


                double e_ik[3], e_jk[3];
                for(int c=0; c<3; c++){
                    e_ik[c] = (atomic_positions_u(i,c) - atomic_positions_u(y,c)) / r_ik;
                    e_jk[c] = (atomic_positions_u(x,c) - atomic_positions_u(y,c)) / r_jk;
                }

                // Determine the location for this triplet of species
                int its;
                if (index_j >= index_k) its = (index_j*(index_j+1))/2 + index_k;
                else its = (index_k*(index_k+1))/2 + index_j;

                offset = nTypes * (1+nG2+nG3);         // Skip this atoms G1 G2 and G3
                offset += its * (nG4+nG5);              // skip G4 and G5 types that are not the ones of atom bi
                o0 = offset;

                if(return_descriptor){
                    // Compute G4
                    computeG4_pyarray(descriptor_mu, idxi, offset, costheta, r_jk, r_ij_square, r_ik_square, r_jk_square, fc_ij, fc_ik);
                    // Compute G5
                    computeG5_pyarray(descriptor_mu, idxi, offset, costheta, r_ij_square, r_ik_square, fc_ij, fc_ik);
                }

                double dcostheta, ang, dang, first_term, secnd_term, third_term, final;
                double d_fc_ik = -(PI*sin((PI*r_ik)/rCut))/(2.0*rCut); // deriv of cutoff function w.r.t. ri
                double d_fc_jk = -(PI*sin((PI*r_jk)/rCut))/(2.0*rCut); // deriv of cutoff function w.r.t. rj


                // grad G4
                if (r_jk <= rCut){
                    for (auto params : g4Params){ // loop over functions

                        eta = params[0];
                        zeta = params[1];
                        lambda = params[2];

                        ef = exp(-eta*(r_ij_square+r_ik_square+r_jk_square));
                        ang = 2*pow(0.5*(1 + lambda*costheta), zeta);
                        dang = zeta*pow(0.5*(1 + lambda*costheta), zeta-1)*lambda;


                        for(int idxj=0; idxj<grad_centers.shape(0); idxj++){
                            int j = grad_centers_u(idxj);
                            if(j!=i && j!=x && j!=y) continue; // the grad center has to be one of the i,x,y or it would not contribute to the gradient here!

                            for(int c=0;c<3;c++) {
                                
                                if(j == i){

                                    dcostheta = 0.5*(r_ik*(r_ij_square-r_ik_square+r_jk_square)*e_ij[c] - r_ij*(r_ij_square-r_ik_square-r_jk_square)*e_ik[c])/(r_ij_square*r_ik_square);
                                    secnd_term = -2.0*eta * (r_ij*e_ij[c] + r_ik*e_ik[c])*ang*fc4;
                                    third_term = e_ij[c]*d_fc_ij*fc_ik*fc_jk + e_ik[c]*d_fc_ik*fc_ij*fc_jk;
                                    //printf("grad ang[i] %i %i %i [%i]",i,x,y,j);

                                } else if(j == x){

                                    dcostheta = -0.5*(e_ik[c]*r_ij*r_ik - e_ij[c]*r_ik_square + e_jk[c]*r_ij*r_jk + e_ij[c]*r_jk_square)/(r_ij_square * r_ik);
                                    secnd_term = 2.0*eta * (r_ij*e_ij[c] - r_jk*e_jk[c])*ang*fc4;
                                    third_term = -e_ij[c]*d_fc_ij*fc_ik*fc_jk + e_jk[c]*d_fc_jk*fc_ij*fc_ik;
                                    //printf("grad ang[x] %i %i %i [%i]",i,x,y,j);

                                } else if(j == y){

                                    dcostheta = 0.5*(e_ik[c]*r_ij_square - r_ik*(e_ij[c]*r_ij - e_jk[c]*r_jk) - e_ik[c]*r_jk_square)/(r_ij*r_ik_square);
                                    secnd_term = 2.0*eta * (r_ik*e_ik[c] + r_jk*e_jk[c])*ang*fc4;
                                    third_term = -e_ik[c]*d_fc_ik*fc_ij*fc_jk - e_jk[c]*d_fc_jk*fc_ij*fc_ik;
                                    //printf("grad ang[y] %i %i %i [%i]",i,x,y,j);
                                }

                                first_term = dang*dcostheta*fc4;
                                third_term *= ang;

                                //printf(" contribs: %lf[dcostheta] %lf %lf %lf --- ", dcostheta, first_term*ef, secnd_term*ef, third_term*ef);

                                final = first_term + secnd_term + third_term;
                                final *= ef;
                                //printf(" output %lf\n",final);
                                derivatives_mu(idxi,idxj,c,o0) += final;
                            }
                        }
                        o0++;
                    }
                }else{
                    o0 += g4Params.size();
                }


                // grad G5
                for (auto params : g5Params) {
                    eta = params[0];
                    zeta = params[1];
                    lambda = params[2];

                    ef = exp(-eta*(r_ij_square+r_ik_square));
                    ang = 2*pow(0.5*(1 + lambda*costheta), zeta);
                    dang = zeta*pow(0.5*(1 + lambda*costheta), zeta-1)*lambda;


                    for(int idxj=0; idxj<grad_centers.shape(0); idxj++){
                        int j = grad_centers_u(idxj);
                        if(j!=i && j!=x && j!=y) continue; // the grad center has to be one of the i,x,y or it would not contribute to the gradient here!

                        for(int c=0;c<3;c++) {
                            
                            if(j == i){

                                dcostheta = 0.5*(r_ik*(r_ij_square-r_ik_square+r_jk_square)*e_ij[c] - r_ij*(r_ij_square-r_ik_square-r_jk_square)*e_ik[c])/(r_ij_square*r_ik_square);
                                secnd_term = -2.0*eta * (r_ij*e_ij[c] + r_ik*e_ik[c])*ang*fc5;
                                third_term = e_ij[c]*d_fc_ij*fc_ik + e_ik[c]*d_fc_ik*fc_ij;

                            } else if(j == x){

                                dcostheta = dcostheta = -0.5*(e_ik[c]*r_ij*r_ik - e_ij[c]*r_ik_square + e_jk[c]*r_ij*r_jk + e_ij[c]*r_jk_square)/(r_ij_square * r_ik);
                                secnd_term = 2.0*eta * (r_ij*e_ij[c])*ang*fc5;
                                third_term = -e_ij[c]*d_fc_ij*fc_ik;

                            } else if(j == y){

                                dcostheta = 0.5*(e_ik[c]*r_ij_square - r_ik*(e_ij[c]*r_ij - e_jk[c]*r_jk) - e_ik[c]*r_jk_square)/(r_ij*r_ik_square);
                                secnd_term = 2.0*eta * (r_ik*e_ik[c])*ang*fc5;
                                third_term = -e_ik[c]*d_fc_ik*fc_ij;
                            }

                            first_term = dang*dcostheta*fc5;
                            third_term *= ang;

                            final = first_term + secnd_term + third_term;
                            final *= ef;
                            derivatives_mu(idxi,idxj,c,o0) += final;
                        }

                    }
                    o0++;
                }


            }
            
        }
    }
}
