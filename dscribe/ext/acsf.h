#ifndef ACSF_H
#define ACSF_H

#include <unordered_map>
#include <vector>

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

using namespace std;


/**
 * Implementation for the performance-critical parts of ACSF.
 */
class ACSF {

    public:
        ACSF() {};
        ACSF(
            double rCut,
            vector<vector<double> > g2Params,
            vector<double> g3Params,
            vector<vector<double> > g4Params,
            vector<vector<double> > g5Params,
            vector<int> atomicNumbers
        );

        vector<vector<double> > create(vector<vector<double> > &positions, vector<int> &atomicNumbers, const vector<vector<double> > &distances, const vector<vector<int> > &neighbours, vector<int> &indices);
        void setRCut(double rCut);
        void setG2Params(vector<vector<double> > g2Params);
        void setG3Params(vector<double> g3Params);
        void setG4Params(vector<vector<double> > g4Params);
        void setG5Params(vector<vector<double> > g5Params);
        void setAtomicNumbers(vector<int> atomicNumbers);

        double getRCut();
        vector<vector<double> > getG2Params();
        vector<double> getG3Params();
        vector<vector<double> > getG4Params();
        vector<vector<double> > getG5Params();
        vector<int> getAtomicNumbers();
        int nTypes;
        int nTypePairs;
        int nG2;
        int nG3;
        int nG4;
        int nG5;
        double rCut;
        vector<vector<double> > g2Params;
        vector<double> g3Params;
        vector<vector<double> > g4Params;
        vector<vector<double> > g5Params;
        vector<int> atomicNumbers;

    private:
        double computeCutoff(double r_ij);
        void computeG1(vector<double> &output, int &offset, double &fc_ij);
        void computeG2(vector<double> &output, int &offset, double &r_ij, double &fc_ij);
        void computeG3(vector<double> &output, int &offset, double &r_ij, double &fc_ij);
        void computeG4(vector<double> &output, int &offset, double &costheta, double &r_jk, double &r_ij_square, double &r_ik_square, double &r_jk_square, double &fc_ij, double &fc_ik);
        void computeG5(vector<double> &output, int &offset, double &costheta, double &r_ij_square, double &r_ik_square, double &fc_ij, double &fc_ik);
        unordered_map<int, int> atomicNumberToIndexMap;
};

#endif
