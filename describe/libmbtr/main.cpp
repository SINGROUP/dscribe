#include "cmbtr.h"
#include <iostream>
using namespace std;

int main() {
    unsigned int nAtoms = 3;
    vector<vector<float> > positions(nAtoms, vector<float>(3));
    vector<int> atomicNumbers(nAtoms);
    int cellLimit = 3;

    positions[0] = vector<float>(3, 1);
    positions[1] = vector<float>(3, 2);
    positions[2] = vector<float>(3, 3);

    CMBTR mbtr(positions, atomicNumbers, cellLimit);
    mbtr.getDisplacementTensor();
}
