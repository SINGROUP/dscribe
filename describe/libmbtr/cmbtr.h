#ifndef CMBTR_H
#define CMBTR_H

#include <vector>
#include <map>
#include <utility>
using namespace std;

class CMBTR {

    public:
        // Constructors
        CMBTR(vector<vector<float> > positions, vector<int> atomicNumbers, map<int,int> atomicNumberToIndexMap, int cellLimit);

        // Functions
        vector<vector<vector<float> > > getDisplacementTensor();
        vector<vector<float> > getDistanceMatrix();
        vector<vector<float> > getInverseDistanceMatrix();
        map<pair<int, int>, vector<float> > getInverseDistanceMap();

    private:
        // Attributes
        vector<vector<float> > positions;
        vector<int> atomicNumbers;
        map<int,int> atomicNumberToIndexMap;
        int cellLimit;
};

#endif
