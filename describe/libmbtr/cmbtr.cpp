#include "cmbtr.h"
#include <vector>
using namespace std;

CMBTR::CMBTR(vector<vector<float> > positions, vector<int> atomicNumbers, int cellLimit)
    : positions(positions)
    , atomicNumbers(atomicNumbers)
    , cellLimit(cellLimit)
{
}

vector<vector<vector<float> > > CMBTR::getDisplacementTensor()
{
    unsigned int nAtoms = this->atomicNumbers.size();

    // Initialize tensor
    vector<vector<vector<float> > > tensor(nAtoms, vector<vector<float> >(nAtoms, vector<float>(3)));

    // Calculate the distance between all pairs and store in the tensor
    for (unsigned int i=0; i < nAtoms; ++i) {
        for (unsigned int j=0; j < nAtoms; ++j) {

            // Only values of the upper triangle are calculated
            if (i <= j) {
                continue;
            }

            // Calculate distance between the two atoms, store in tensor
            vector<float>& iPos = this->positions[i];
            vector<float>& jPos = this->positions[i];
            vector<float> diff(3);
            vector<float> negDiff(3);

            for (unsigned int k=0; k < 3; ++k) {
                float iDiff = iPos[k] - jPos[k];
                diff[k] = iDiff;
                negDiff[k] = -iDiff;
            }

            tensor[i][j] = diff;
            tensor[j][i] = negDiff;
        }
    }

    return tensor;
}
