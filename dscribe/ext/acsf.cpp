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
                    if (k >= j) {
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
