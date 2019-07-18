#ifndef ACSF_H
#define ACSF_H

#include <unordered_map>

#define PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

using namespace std;


/**
 * Implementation for the performance-critical parts of ACSF.
 */
class ACSF {

    public:
        ACSF() {};
        ACSF(
            float rCut,
            vector<vector<float> > g2Params,
            vector<float> g3Params,
            vector<vector<float> > g4Params,
            vector<vector<float> > g5Params,
            vector<int> atomicNumbers
        );

        vector<vector<float> > create(vector<vector<float> > &positions, vector<int> &atomicNumbers, const vector<vector<float> > &distances, const vector<vector<int> > &neighbours, vector<int> &indices);
        void setRCut(float rCut);
        void setG2Params(vector<vector<float> > g2Params);
        void setG3Params(vector<float> g3Params);
        void setG4Params(vector<vector<float> > g4Params);
        void setG5Params(vector<vector<float> > g5Params);
        void setAtomicNumbers(vector<int> atomicNumbers);

        int nTypes;
        int nTypePairs;
        int nG2;
        int nG3;
        int nG4;
        int nG5;
        float rCut;
        vector<vector<float> > g2Params;
        vector<float> g3Params;
        vector<vector<float> > g4Params;
        vector<vector<float> > g5Params;
        vector<int> atomicNumbers;

    private:
        float computeCutoff(float r_ij);
        void computeG1(vector<float> &output, int &offset, float &fc_ij);
        void computeG2(vector<float> &output, int &offset, float &r_ij, float &fc_ij);
        void computeG3(vector<float> &output, int &offset, float &r_ij, float &fc_ij);
        void computeG4(vector<float> &output, int &offset, float &costheta, float &r_jk, float &r_ij_square, float &r_ik_square, float &r_jk_square, float &fc_ij, float &fc_ik);
        void computeG5(vector<float> &output, int &offset, float &costheta, float &r_ij_square, float &r_ik_square, float &fc_ij, float &fc_ik);
        unordered_map<int, int> atomicNumberToIndexMap;
};

#endif
