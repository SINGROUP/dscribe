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

ACSF::ACSF(double r_cut, vector<vector<double> > g2_params, vector<double> g3_params, vector<vector<double> > g4_params, vector<vector<double> > g5_params, vector<int> atomic_numbers, bool periodic)
    : DescriptorLocal(periodic, "off", r_cut)
{
    set_r_cut(r_cut);
    set_g2_params(g2_params);
    set_g3_params(g3_params);
    set_g4_params(g4_params);
    set_g5_params(g5_params);
    set_atomic_numbers(atomic_numbers);
}

void ACSF::set_r_cut(double r_cut)
{
    this->r_cut = r_cut;
}
double ACSF::get_r_cut()
{
return this->r_cut;
}

void ACSF::set_g2_params(vector<vector<double> > g2_params)
{
    this->g2_params = g2_params;
    n_g2 = g2_params.size();
}

vector<vector<double> > ACSF::get_g2_params()
{
    return this->g2_params;
}


void ACSF::set_g3_params(vector<double> g3_params)
{
    this->g3_params = g3_params;
    n_g3 = g3_params.size();
}
vector<double> ACSF::get_g3_params()
{
    return this->g3_params;
}


void ACSF::set_g4_params(vector<vector<double> > g4_params)
{
    this->g4_params = g4_params;
    n_g4 = g4_params.size();
}
vector<vector<double> > ACSF::get_g4_params()
{
    return this->g4_params;
}


void ACSF::set_g5_params(vector<vector<double> > g5_params)
{
    this->g5_params = g5_params;
    n_g5 = g5_params.size();
}
vector<vector<double> > ACSF::get_g5_params()
{
    return this->g5_params;
}


void ACSF::set_atomic_numbers(vector<int> atomic_numbers)
{
    this->atomic_numbers = atomic_numbers;
    n_types = atomic_numbers.size();
    n_type_pairs = n_types * (n_types + 1) / 2;
    unordered_map<int, int> atomic_number_to_index_map;
    int i = 0;
    for (int Z : atomic_numbers) {
        atomic_number_to_index_map[Z] = i;
        ++i;
    }
    this->atomic_number_to_index_map = atomic_number_to_index_map;
}
vector<int> ACSF::get_atomic_numbers()
{
    return this->atomic_numbers;
}

inline int ACSF::get_number_of_features() const
{
    return (1 + this->n_g2 + this->n_g3) * this->n_types + (this->n_g4 + this->n_g5) * this->n_type_pairs;
}

void ACSF::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> centers,
    CellList cell_list
)
{
    // This overload is likely for Python calls where centers are doubles,
    // but the actual implementation expects int indices for centers.
    // It's left empty as per the original acsf_new.cpp.
}

void ACSF::create(
    py::array_t<double> out, 
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<int> centers,
    CellList cell_list
)
{
    auto out_mu = out.mutable_unchecked<2>();
    auto centers_u = centers.unchecked<1>(); 
    auto positions_u = positions.unchecked<2>(); 
    auto atomic_numbers_u = atomic_numbers.unchecked<1>(); 

    // Loop through centers
    const int n_centers = centers.shape(0);
    for (int index_i = 0; index_i < n_centers; ++index_i) {
        int i = centers_u[index_i];

        // Loop through neighbours
        CellListResult neighbours_i = cell_list.getNeighboursForIndex(i);
        int n_neighbours = neighbours_i.indices.size();
        for (int j_neighbour = 0; j_neighbour < n_neighbours; ++j_neighbour) {
            int j = neighbours_i.indices[j_neighbour];

            // Precompute some values
            double r_ij = neighbours_i.distances[j_neighbour];
            double fc_ij = compute_cutoff(r_ij);
            int index_j = atomic_number_to_index_map[atomic_numbers_u[j]];
            int offset = index_j * (1+n_g2+n_g3);  // Skip G1, G2, G3 types that are not the ones of atom bi

            // Compute G1
            compute_g1(out_mu, index_i, offset, fc_ij);

            // Compute G2
            compute_g2(out_mu, index_i, offset, r_ij, fc_ij);

            // Compute G3
            compute_g3(out_mu, index_i, offset, r_ij, fc_ij);

            // If g4 or g5 requested, loop through second neighbours
            if (g4_params.size() != 0 || g5_params.size() != 0) {
                for (int k_neighbour = 0; k_neighbour < n_neighbours; ++k_neighbour) {
                    int k = neighbours_i.indices[k_neighbour];
                    if (k >= j) { // Ensure k < j to avoid duplicate triplets and self-interaction
                        continue;
                    }

                    // Calculate j-k distance: it is not contained in the cell lists.
                    double dx = positions_u(j, 0) - positions_u(k, 0);
                    double dy = positions_u(j, 1) - positions_u(k, 1);
                    double dz = positions_u(j, 2) - positions_u(k, 2);
                    double r_jk_square = dx*dx + dy*dy + dz*dz;
                    double r_jk = sqrt(r_jk_square);

                    // Precompute some values that are used by both G4 and G5
                    double r_ik = neighbours_i.distances[k_neighbour];
                    double fc_ik = compute_cutoff(r_ik);
                    double r_ij_square = neighbours_i.distancesSquared[j_neighbour];
                    double r_ik_square = neighbours_i.distancesSquared[k_neighbour];
                    int index_k = atomic_number_to_index_map[atomic_numbers_u[k]];
                    double costheta = 0.5/(r_ij*r_ik) * (r_ij_square+r_ik_square-r_jk_square);

                    // Determine the location for this triplet of species
                    int its;
                    if (index_j >= index_k) {
                        its = (index_j*(index_j+1))/2 + index_k;
                    } else  {
                        its = (index_k*(index_k+1))/2 + index_j;
                    }
                    offset = n_types * (1+n_g2+n_g3);         // Skip this atoms G1 G2 and G3
                    offset += its * (n_g4+n_g5);              // Skip G4 and G5 types that are not the ones of atom bi

                    // Compute G4
                    compute_g4(out_mu, index_i, offset, costheta, r_jk, r_ij_square, r_ik_square, r_jk_square, fc_ij, fc_ik);

                    // Compute G5
                    compute_g5(out_mu, index_i, offset, costheta, r_ij_square, r_ik_square, fc_ij, fc_ik);
                }
            }
        }
    }
}

/*! \brief Computes the value of the cutoff fuction at a specific distance.
 * */
inline double ACSF::compute_cutoff(double r_ij) {
	return 0.5 * (cos(r_ij * PI / r_cut) + 1);
}

inline void ACSF::compute_g1(py::detail::unchecked_mutable_reference<double, 2> &out_mu, int &index, int &offset, double &fc_ij) {
    out_mu(index, offset) += fc_ij;
    offset += 1;
}

inline void ACSF::compute_g2(py::detail::unchecked_mutable_reference<double, 2> &out_mu, int &index, int &offset, double &r_ij, double &fc_ij) {
    double eta;
    double Rs;
	for (auto params : g2_params) {
        eta = params[0];
        Rs = params[1];
        out_mu(index, offset) += exp(-eta * (r_ij - Rs)*(r_ij - Rs)) * fc_ij;
        offset++;
	}
}

inline void ACSF::compute_g3(py::detail::unchecked_mutable_reference<double, 2> &out_mu, int &index, int &offset, double &r_ij, double &fc_ij) {
	for (auto param : g3_params) {
        out_mu(index, offset) += cos(r_ij*param)*fc_ij;
        offset++;
    }
}

inline void ACSF::compute_g4(py::detail::unchecked_mutable_reference<double, 2> &out_mu, int &index, int &offset, double &costheta, double &r_jk, double &r_ij_square, double &r_ik_square, double &r_jk_square, double &fc_ij, double &fc_ik) {
    if (r_jk > r_cut) {
        offset += g4_params.size();
        return;
    }
    double cutoff_jk = compute_cutoff(r_jk);
	double fc4 = fc_ij*fc_ik*cutoff_jk;
	double eta;
	double zeta;
	double lambda;
	double gauss;
	for (auto params : g4_params) {
		eta = params[0];
		zeta = params[1];
		lambda = params[2];
		gauss = exp(-eta*(r_ij_square+r_ik_square+r_jk_square)) * fc4;
		out_mu(index, offset) += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
		offset++;
	}
}

inline void ACSF::compute_g5(py::detail::unchecked_mutable_reference<double, 2> &out_mu, int &index, int &offset, double &costheta, double &r_ij_square, double &r_ik_square, double &fc_ij, double &fc_ik) {
	double eta;
	double zeta;
	double lambda;
	double gauss;
	double fc5 = fc_ij*fc_ik;
	for (auto params : g5_params) {
		eta = params[0];
		zeta = params[1];
		lambda = params[2];
		gauss = exp(-eta*(r_ij_square+r_ik_square)) * fc5;
		out_mu(index, offset) += 2*pow(0.5*(1 + lambda*costheta), zeta) * gauss;
		offset++;
	}
}


void ACSF::derivatives_analytical(
    py::array_t<double> derivatives,
    py::array_t<double> descriptor,
    py::array_t<int> atomic_numbers,
    py::array_t<double> atomic_positions,
    CellList cell_list,
    py::array_t<int> desc_centers,
    py::array_t<int> grad_centers, // we want derivatives w.r.t. these atoms
    const bool return_descriptor
) {
    int n_desc_centers = desc_centers.shape(0);

    auto descriptor_mu = descriptor.mutable_unchecked<2>(); // [n_desc_centers, n_features]
    auto derivatives_mu = derivatives.mutable_unchecked<4>(); // [n_desc_centers, n_grad_centers, 3, n_features]
    
    auto atomic_numbers_u = atomic_numbers.unchecked<1>();
    auto atomic_positions_u = atomic_positions.unchecked<2>();

    auto desc_centers_u = desc_centers.unchecked<1>();
    auto grad_centers_u = grad_centers.unchecked<1>();

    // d[idxi,idxj,c,l] = derivative of l-th ACSF of atom desc_centers_u(idxi), w.r.t. atom grad_centers_u(idxj) coordinate c
    // descriptor_mu(idxi, l) = l-th ACSF of atom desc_centers_u(idxi)

    // Declare eta, zeta, lambda here to make them visible in the loops
    double eta, zeta, lambda; 

    for(int idxi=0; idxi<n_desc_centers; idxi++) {
        
        int i = desc_centers_u(idxi);
        // Get neighbours and distances for atom i using CellList
        CellListResult neighbours_i = cell_list.getNeighboursForIndex(i);
        int n_neighbours = neighbours_i.indices.size();

        // Loop over neighbours of atom i (j)
        for (int j_neighbour_idx = 0; j_neighbour_idx < n_neighbours; ++j_neighbour_idx) {
            int x = neighbours_i.indices[j_neighbour_idx]; // x is the j atom
            if(i == x) continue; // Skip self-interaction

            // Precompute some values for G1, G2, G3
            double r_ij = neighbours_i.distances[j_neighbour_idx]; // Corrected access
            double r_ij_square = neighbours_i.distancesSquared[j_neighbour_idx]; // Add this for G4/G5 later
            double fc_ij = compute_cutoff(r_ij);
            int index_j = atomic_number_to_index_map[atomic_numbers_u[x]];
            // Initial offset for G1, G2, G3 for the current atomic species (index_j)
            int offset = index_j * (1+n_g2+n_g3);  
            int o0_g1_g3 = offset; // Store initial offset for derivatives calculation

            // Compute unit vector e_ij = (pos_i - pos_j) / r_ij
            double e_ij[3];
            for(int c=0; c<3; c++)
                e_ij[c] = (atomic_positions_u(i,c) - atomic_positions_u(x,c)) / r_ij;
            
            // If return_descriptor is true, compute and add descriptor values
            if(return_descriptor){
                // Note: The compute_gX functions in acsf_new.cpp take out_mu, index, offset.
                // Here, out_mu is descriptor_mu, index is idxi (the index in the descriptor array),
                // and offset is the current feature offset.
                int current_offset_for_descriptor = offset; // Use a temporary offset for descriptor calculation
                compute_g1(descriptor_mu, idxi, current_offset_for_descriptor, fc_ij);
                compute_g2(descriptor_mu, idxi, current_offset_for_descriptor, r_ij, fc_ij);
                compute_g3(descriptor_mu, idxi, current_offset_for_descriptor, r_ij, fc_ij);
            }

            // Derivative of cutoff function w.r.t. r_ij
            double d_fc_ij = -(PI*sin((PI*r_ij)/r_cut))/(2.0*r_cut);

            // G1 Derivatives
            for(int idxj=0; idxj<grad_centers.shape(0); idxj++){
                int j = grad_centers_u(idxj);
                if(j != i && j != x) continue; // Only i and x contribute to G1 derivative
                if(j == i){
                    for(int c=0;c<3;c++)
                        derivatives_mu(idxi,idxj,c,o0_g1_g3) += e_ij[c]*d_fc_ij;
                }
                else if(j == x){
                    for(int c=0;c<3;c++)
                        derivatives_mu(idxi,idxj,c,o0_g1_g3) -= e_ij[c]*d_fc_ij;
                }
            }
            o0_g1_g3++; // Move to next feature for G2

            // G2 Derivatives
            double Rs, ef, der, tmp;
            for (auto params : g2_params) {
                eta = params[0]; // eta is declared outside the loop now
                Rs = params[1];
                tmp = (r_ij - Rs);
                ef = exp(-eta * tmp*tmp);
                der = ef*(-eta*2*tmp*fc_ij + d_fc_ij); // Derivative w.r.t r_ij

                for(int idxj=0; idxj<grad_centers.shape(0); idxj++){
                    int j = grad_centers_u(idxj);
                    if(j != i && j != x) continue;
                    if(j == i){
                        for(int c=0;c<3;c++)
                            derivatives_mu(idxi,idxj,c,o0_g1_g3) += e_ij[c]*der;
                    }
                    else if(j==x){
                        for(int c=0;c<3;c++)
                            derivatives_mu(idxi,idxj,c,o0_g1_g3) -= e_ij[c]*der;
                    }
                }
                o0_g1_g3++; // Move to next feature for G3
            }

            // G3 Derivatives
            for (auto param : g3_params) {
                der = param * (-sin(r_ij*param)*fc_ij) + cos(r_ij*param)*d_fc_ij; // Derivative w.r.t r_ij

                for(int idxj=0; idxj<grad_centers.shape(0); idxj++){
                    int j = grad_centers_u(idxj);
                    if(j != i && j != x) continue;
                    if(j == i){
                        for(int c=0;c<3;c++)
                            derivatives_mu(idxi,idxj,c,o0_g1_g3) += e_ij[c]*der;
                    }
                    else if(j==x){
                        for(int c=0;c<3;c++)
                            derivatives_mu(idxi,idxj,c,o0_g1_g3) -= e_ij[c]*der;
                    }
                }
                o0_g1_g3++; // Move to next feature (if any, for G4/G5)
            }
            
            // If no G4 or G5 parameters, continue to next j_neighbour
            if (g4_params.size()==0 && g5_params.size()==0) continue;
            
            // Loop over second neighbours of atom i (k or y) for 3-body terms (G4, G5)
            for (int k_neighbour_idx = 0; k_neighbour_idx < n_neighbours; ++k_neighbour_idx) {
                int y = neighbours_i.indices[k_neighbour_idx]; // y is the k atom
                if (y == i || k_neighbour_idx >= j_neighbour_idx) { // Ensure k index is less than j index to avoid duplicates (and self-k)
                    continue;
                }

                // Calculate j-k distance: it is not contained in the cell lists.
                double dx = atomic_positions_u(x, 0) - atomic_positions_u(y, 0);
                double dy = atomic_positions_u(x, 1) - atomic_positions_u(y, 1);
                double dz = atomic_positions_u(x, 2) - atomic_positions_u(y, 2);
                double r_jk_square = dx*dx + dy*dy + dz*dz;
                double r_jk = sqrt(r_jk_square);

                // Precompute values for G4 and G5
                double r_ik = neighbours_i.distances[k_neighbour_idx]; // Corrected access
                double fc_ik = compute_cutoff(r_ik);
                double r_ik_square = neighbours_i.distancesSquared[k_neighbour_idx]; // Corrected access
                // r_ij_square is already defined from the outer loop
                int index_k = atomic_number_to_index_map[atomic_numbers_u[y]];
                double costheta = 0.5*(r_ij_square+r_ik_square-r_jk_square)/(r_ij*r_ik);
                double fc4 = fc_ij*fc_ik*compute_cutoff(r_jk); // G4 cutoff product, compute_cutoff for r_jk
                double fc5 = fc_ij*fc_ik;       // G5 cutoff product

                // Unit vectors for i-k and j-k
                double e_ik[3], e_jk[3];
                for(int c=0; c<3; c++){
                    e_ik[c] = (atomic_positions_u(i,c) - atomic_positions_u(y,c)) / r_ik;
                    e_jk[c] = (atomic_positions_u(x,c) - atomic_positions_u(y,c)) / r_jk;
                }

                // Determine the location for this triplet of species
                int its;
                if (index_j >= index_k) its = (index_j*(index_j+1))/2 + index_k;
                else its = (index_k*(index_k+1))/2 + index_j;

                // Offset for G4 and G5 features
                offset = n_types * (1+n_g2+n_g3);         // Skip G1, G2, G3 features for all types
                offset += its * (n_g4+n_g5);              // Skip G4 and G5 features for other type pairs
                int o0_g4_g5 = offset; // Store initial offset for derivatives calculation

                // If return_descriptor is true, compute and add descriptor values
                if(return_descriptor){
                    int current_offset_for_descriptor = offset; // Use a temporary offset for descriptor calculation
                    compute_g4(descriptor_mu, idxi, current_offset_for_descriptor, costheta, r_jk, r_ij_square, r_ik_square, r_jk_square, fc_ij, fc_ik);
                    compute_g5(descriptor_mu, idxi, current_offset_for_descriptor, costheta, r_ij_square, r_ik_square, fc_ij, fc_ik);
                }

                // Initialize variables to prevent uninitialized warnings
                double dcostheta = 0.0;
                double ang = 0.0;
                double dang = 0.0;
                double first_term = 0.0;
                double secnd_term = 0.0;
                double third_term = 0.0;
                double final = 0.0;

                // Derivatives of cutoff functions w.r.t. r_ik and r_jk
                double d_fc_ik = -(PI*sin((PI*r_ik)/r_cut))/(2.0*r_cut);
                double d_fc_jk = -(PI*sin((PI*r_jk)/r_cut))/(2.0*r_cut);

                // G4 Derivatives
                if (r_jk <= r_cut){ // Only compute if r_jk is within cutoff for G4
                    for (auto params : g4_params){ // loop over G4 functions
                        eta = params[0]; // eta is declared outside the loop now
                        zeta = params[1]; // zeta is declared outside the loop now
                        lambda = params[2]; // lambda is declared outside the loop now

                        ef = exp(-eta*(r_ij_square+r_ik_square+r_jk_square));
                        ang = 2*pow(0.5*(1 + lambda*costheta), zeta);
                        dang = zeta*pow(0.5*(1 + lambda*costheta), zeta-1)*lambda;

                        for(int idxj=0; idxj<grad_centers.shape(0); idxj++){
                            int j = grad_centers_u(idxj);
                            // Only i, x, y contribute to the gradient of G4/G5 for this triplet
                            if(j!=i && j!=x && j!=y) continue; 

                            for(int c=0;c<3;c++) {
                                // Calculate dcostheta and other terms based on which atom (i, x, or y) is the gradient center
                                if(j == i){ // Derivative w.r.t. atom i
                                    dcostheta = 0.5*(r_ik*(r_ij_square-r_ik_square+r_jk_square)*e_ij[c] - r_ij*(r_ij_square-r_ik_square-r_jk_square)*e_ik[c])/(r_ij_square*r_ik_square);
                                    secnd_term = -2.0*eta * (r_ij*e_ij[c] + r_ik*e_ik[c])*ang*fc4;
                                    third_term = (e_ij[c]*d_fc_ij*fc_ik*compute_cutoff(r_jk) + e_ik[c]*d_fc_ik*fc_ij*compute_cutoff(r_jk)) * ang + e_jk[c]*d_fc_jk*fc_ij*fc_ik * ang;

                                } else if(j == x){ // Derivative w.r.t. atom x
                                    dcostheta = -0.5*(e_ik[c]*r_ij*r_ik - e_ij[c]*r_ik_square + e_jk[c]*r_ij*r_jk + e_ij[c]*r_jk_square)/(r_ij_square * r_ik);
                                    secnd_term = 2.0*eta * (r_ij*e_ij[c] - r_jk*e_jk[c])*ang*fc4;
                                    third_term = (-e_ij[c]*d_fc_ij*fc_ik*compute_cutoff(r_jk) + e_jk[c]*d_fc_jk*fc_ij*fc_ik) * ang;

                                } else if(j == y){ // Derivative w.r.t. atom y
                                    dcostheta = 0.5*(e_ik[c]*r_ij_square - r_ik*(e_ij[c]*r_ij - e_jk[c]*r_jk) - e_ik[c]*r_jk_square)/(r_ij*r_ik_square);
                                    secnd_term = 2.0*eta * (r_ik*e_ik[c] + r_jk*e_jk[c])*ang*fc4;
                                    third_term = (-e_ik[c]*d_fc_ik*fc_ij*compute_cutoff(r_jk) - e_jk[c]*d_fc_jk*fc_ij*fc_ik) * ang;
                                }

                                first_term = dang*dcostheta*fc4;
                                final = (first_term + secnd_term + third_term) * ef;
                                derivatives_mu(idxi,idxj,c,o0_g4_g5) += final;
                            }
                        }
                        o0_g4_g5++; // Move to next G4 feature
                    }
                } else {
                    o0_g4_g5 += g4_params.size(); // Skip G4 features if r_jk is outside cutoff
                }

                // G5 Derivatives
                for (auto params : g5_params) {
                    eta = params[0]; // eta is declared outside the loop now
                    zeta = params[1]; // zeta is declared outside the loop now
                    lambda = params[2]; // lambda is declared outside the loop now

                    ef = exp(-eta*(r_ij_square+r_ik_square));
                    ang = 2*pow(0.5*(1 + lambda*costheta), zeta);
                    dang = zeta*pow(0.5*(1 + lambda*costheta), zeta-1)*lambda;

                    for(int idxj=0; idxj<grad_centers.shape(0); idxj++){
                        int j = grad_centers_u(idxj);
                        if(j!=i && j!=x && j!=y) continue; 

                        for(int c=0;c<3;c++) {
                            
                            if(j == i){ // Derivative w.r.t. atom i
                                dcostheta = 0.5*(r_ik*(r_ij_square-r_ik_square+r_jk_square)*e_ij[c] - r_ij*(r_ij_square-r_ik_square-r_jk_square)*e_ik[c])/(r_ij_square*r_ik_square);
                                secnd_term = -2.0*eta * (r_ij*e_ij[c] + r_ik*e_ik[c])*ang*fc5;
                                third_term = (e_ij[c]*d_fc_ij*fc_ik + e_ik[c]*d_fc_ik*fc_ij) * ang;

                            } else if(j == x){ // Derivative w.r.t. atom x
                                dcostheta = -0.5*(e_ik[c]*r_ij*r_ik - e_ij[c]*r_ik_square + e_jk[c]*r_ij*r_jk + e_ij[c]*r_jk_square)/(r_ij_square * r_ik);
                                secnd_term = 2.0*eta * (r_ij*e_ij[c])*ang*fc5;
                                third_term = (-e_ij[c]*d_fc_ij*fc_ik) * ang;

                            } else if(j == y){ // Derivative w.r.t. atom y
                                dcostheta = 0.5*(e_ik[c]*r_ij_square - r_ik*(e_ij[c]*r_ij - e_jk[c]*r_jk) - e_ik[c]*r_jk_square)/(r_ij*r_ik_square);
                                secnd_term = 2.0*eta * (r_ik*e_ik[c])*ang*fc5;
                                third_term = (-e_ik[c]*d_fc_ik*fc_ij) * ang;
                            }

                            first_term = dang*dcostheta*fc5;
                            final = (first_term + secnd_term + third_term) * ef;
                            derivatives_mu(idxi,idxj,c,o0_g4_g5) += final;
                        }
                    }
                    o0_g4_g5++; // Move to next G5 feature
                }
            }
        }
    }
}
