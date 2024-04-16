#ifndef ACSF_H
#define ACSF_H

#include <unordered_map>
#include <vector>
#include "descriptorlocal.h"

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

using namespace std;


/**
 * Implementation for the performance-critical parts of ACSF.
 */
class ACSF : public DescriptorLocal {
    public:
        ACSF(
            double r_cut,
            vector<vector<double> > g2_params,
            vector<double> g3_params,
            vector<vector<double> > g4_params,
            vector<vector<double> > g5_params,
            vector<int> atomic_numbers,
            bool periodic
        );

        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<int> centers,
            CellList cell_list
        );

        void create(
            py::array_t<double> out, 
            py::array_t<double> positions,
            py::array_t<int> atomic_numbers,
            py::array_t<double> centers,
            CellList cell_list
        );

        int get_number_of_features() const;

        void set_r_cut(double r_cut);
        void set_g2_params(vector<vector<double> > g2_params);
        void set_g3_params(vector<double> g3_params);
        void set_g4_params(vector<vector<double> > g4_params);
        void set_g5_params(vector<vector<double> > g5_params);
        void set_atomic_numbers(vector<int> atomic_numbers);

        double get_r_cut();
        vector<vector<double> > get_g2_params();
        vector<double> get_g3_params();
        vector<vector<double> > get_g4_params();
        vector<vector<double> > get_g5_params();
        vector<int> get_atomic_numbers();
        int n_types;
        int n_type_pairs;
        int n_g2;
        int n_g3;
        int n_g4;
        int n_g5;
        double r_cut;
        vector<vector<double> > g2_params;
        vector<double> g3_params;
        vector<vector<double> > g4_params;
        vector<vector<double> > g5_params;
        vector<int> atomic_numbers;

    private:
        double compute_cutoff(double r_ij);
        void compute_g1(py::detail::unchecked_mutable_reference<double, 2> &out_mu, int &index, int &offset, double &fc_ij);
        void compute_g2(py::detail::unchecked_mutable_reference<double, 2> &out_mu, int &index, int &offset, double &r_ij, double &fc_ij);
        void compute_g3(py::detail::unchecked_mutable_reference<double, 2> &out_mu, int &index, int &offset, double &r_ij, double &fc_ij);
        void compute_g4(py::detail::unchecked_mutable_reference<double, 2> &out_mu, int &index, int &offset, double &costheta, double &r_jk, double &r_ij_square, double &r_ik_square, double &r_jk_square, double &fc_ij, double &fc_ik);
        void compute_g5(py::detail::unchecked_mutable_reference<double, 2> &out_mu, int &index, int &offset, double &costheta, double &r_ij_square, double &r_ik_square, double &fc_ij, double &fc_ik);
        unordered_map<int, int> atomic_number_to_index_map;
};

#endif
