#ifndef ACSF_H
#define ACSF_H

#include <map>

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

        vector<vector<float> > create(vector<vector<float> > &positions, vector<int> &atomicNumbers, vector<vector<float> > &distances, vector<int> &indices);
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
        float computeCutoff(float Rij);
        void computeBond(vector<float> &output, vector<int> &atomicNumbers, vector<vector<float> > &distances, int ai, int bi);
        void computeAngle(vector<float> &output, vector<int> &atomicNumbers, vector<vector<float> > &distances, int i, int j, int k);
        map<int, int> atomicNumberToIndexMap;
};

#endif
