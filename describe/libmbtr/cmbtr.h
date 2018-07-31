#ifndef CMBTR_H
#define CMBTR_H

#include <vector>
using namespace std;

class CMBTR {

    public:
        // Constructors
        CMBTR(vector<vector<float> > positions, vector<int> atomicNumbers, int cellLimit);

        // Functions
        vector<vector<vector<float> > > getDisplacementTensor();

    private:
        // Attributes
        vector<vector<float> > positions;
        vector<int> atomicNumbers;
        int cellLimit;
};

#endif
