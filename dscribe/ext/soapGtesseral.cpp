/*Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <map>
#include <set>
#include "soapGeneral.h"
#include "celllist.h"

#define tot (double*) malloc(sizeof(double)*totalAN);
#define totrs (double*) malloc(sizeof(double)*totalAN*rsize);
#define sd sizeof(double)
#define PI 3.14159265359


//inline void expMs(double* rExpDiff, double alpha, double r, double ri, int isize)
//{
//    double rDiff;
//    for (int i = 0; i < isize; i++) {
//            rDiff = r - ri;
//            if (rDiff > 5.0 ) {
//                rExpDiff[i] = 0.0;
//            } else {
//                rExpDiff[i] = exp(-alpha*rDiff*rDiff);
//            }
//    }
//}
//inline void expPs(double* rExpSum, double alpha, double r, double ri, int isize)
//{
//    double rSum;
//    for (int i = 0; i < isize; i++) {
////            rSum = r + ri;
//            if (rSum > 5.0 ) {
//                rExpSum[i] = 0.0;
//            } else {
//                rExpSum[i] = exp(-alpha*rSum*rSum);
//            }
//    }
//}
//int getDeltas(double* dx, double* dy, double* dz, double* ri, double* rw, double rCut, double* oOri, double* oO4arri, double* minExp, double* pluExp, int* isCenter, double alpha, const py::array_t<double> &positions, const double ix, const double iy, const double iz, const vector<int> &indices, int rsize, int Ihpos, int Itype)
//{
//    int icount = 0;
//    double ri2;
//    double oOa = 1/alpha;
//    double Xi; double Yi; double Zi;
//    int nNeighbours = indices.size();
//    double* oO4ari = (double*) malloc(sd*nNeighbours);
//
//    auto pos = positions.unchecked<2>();
//    for (const int &i : indices) {
//        Xi = pos(i, 0) - ix;
//        Yi = pos(i, 1) - iy;
//        Zi = pos(i, 2) - iz;
//        ri2 = Xi*Xi + Yi*Yi + Zi*Zi;
//
//        if (ri2<=1e-12) {
//            isCenter[0] = 1;
//        } else {
//            ri[icount] = sqrt(ri2);
//            dx[icount] = Xi;
//            dy[icount] = Yi;
//            dz[icount] = Zi;
//            oOri[icount] = 1/ri[icount];
//            oO4ari[icount] = 0.25*oOa*oOri[icount];
//            icount++;
//        }
//    }
//
//    double* oOr = getoOr(rw, rsize);
//    for (int i = 0; i < icount; i++) {
//        for (int w = 0; w < rsize; w++) {
//            oO4arri[rsize*i + w] = oO4ari[i]*oOr[w];
//        }
//    }
//    expMs(minExp, alpha, rw, ri, icount, rsize);
//    expPs(pluExp, alpha, rw, ri, icount, rsize);
//
//    free(oO4ari);
//    return icount;
//}
  //countMax = isize ----------------------------
//  double* oOr = getoOr(rw, rsize);
//  for(int i = 0; i < icount; i++){
//    for(int w = 0; w < rsize; w++){
//      oO4arri[rsize*i + w] = oO4ari[i]*oOr[w];
//    }
//  }
//  expMs(minExp,alpha,rw,ri,icount,rsize);
//  expPs(pluExp,alpha,rw,ri,icount,rsize);
//
//  free(oO4ari);
//
//  return icount;
//}
void getC(double* Cs, double rCut,int lMax, int rsize, int gnsize,int* isCenter, double alpha)
{
    for (int i = 0; i < 2*(lMax+1)*(lMax+1)*gnsize; i++) {
        Cs[i] = 0.0;
    }

    for (int n = 0; n < gnsize; n++) {
        //for i0 case
        if (isCenter[0]==1) {
            for (int rw = 0; rw < rsize; rw++) {
                Cs[2*(lMax+1)*(lMax+1)*n] += 0.5*0.564189583547756*rw2[rw]*ws[rw]*gns[rsize*n + rw]*exp(-alpha*rw2[rw]);
            }
        }
        for (int l = 0; l < lMax+1; l++) {
            for (int m = 0; m < l+1; m++) {
                for (int rw = 0; rw < rsize; rw++) {
                    Cs[2*(lMax+1)*(lMax+1)*n + l*2*(lMax+1) + 2*m    ] += rw2[rw]*ws[rw]*gns[rsize*n + rw]*summed[2*(lMax+1)*l*rsize + 2*m*rsize + 2*rw    ]; // Re
                    Cs[2*(lMax+1)*(lMax+1)*n + l*2*(lMax+1) + 2*m + 1] += rw2[rw]*ws[rw]*gns[rsize*n + rw]*summed[2*(lMax+1)*l*rsize + 2*m*rsize + 2*rw + 1]; //Im
                }
            }
        }
    }
    isCenter[0] = 0;
}
void accumC(double* Cts, double* Cs, int lMax, int gnsize, int typeI)
{
    for (int n = 0; n < gnsize; n++) {
        for (int l = 0; l < lMax+1; l++) {
            for (int m = 0; m < l+1; m++) {
                Cts[2*typeI*(lMax+1)*(lMax+1)*gnsize +2*(lMax+1)*(lMax+1)*n + l*2*(lMax+1) + 2*m    ] = Cs[2*(lMax+1)*(lMax+1)*n + l*2*(lMax+1) + 2*m    ];
                Cts[2*typeI*(lMax+1)*(lMax+1)*gnsize +2*(lMax+1)*(lMax+1)*n + l*2*(lMax+1) + 2*m + 1] = Cs[2*(lMax+1)*(lMax+1)*n + l*2*(lMax+1) + 2*m + 1];
            }
        }
    }
}
void getPs(double* Ps, double* Cts,  int Nt, int lMax, int gnsize, bool crossover)
{
    int NN = ((gnsize+1)*gnsize)/2;
    int nTypeComb = crossover ? ((Nt+1)*Nt)/2 : Nt;
    int nshift = 0;
    int tshift = 0;
    for (int i = 0; i < nTypeComb*(lMax+1)*NN; i++) {
        Ps[i] = 0.0;
    }

    for (int t1 = 0; t1 < Nt; t1++) {
        int t2Limit = crossover ? Nt : t1+1;
        for (int t2 = t1; t2 < t2Limit; t2++) {
            for (int l = 0; l < lMax+1; l++) {
                nshift = 0;
                for (int n = 0; n < gnsize; n++) {
                    for (int nd = n; nd < gnsize; nd++) {
                        for (int m = 0; m < l+1; m++) {
                            if (m==0) {
                                Ps[tshift*(lMax+1)*NN + l*NN + nshift ]
                                    +=  Cts[2*t1*(lMax+1)*(lMax+1)*gnsize + 2*(lMax+1)*(lMax+1)*n  + l*2*(lMax+1)] // m=0
                                    *Cts[2*t2*(lMax+1)*(lMax+1)*gnsize + 2*(lMax+1)*(lMax+1)*nd + l*2*(lMax+1)]; // m=0
                            } else {
                                Ps[tshift*(lMax+1)*NN + l*NN + nshift]
                                    +=  2*(Cts[2*t1*(lMax+1)*(lMax+1)*gnsize + 2*(lMax+1)*(lMax+1)*n  + l*2*(lMax+1) + 2*m]
                                            *Cts[2*t2*(lMax+1)*(lMax+1)*gnsize + 2*(lMax+1)*(lMax+1)*nd + l*2*(lMax+1) + 2*m]
                                            + Cts[2*t1*(lMax+1)*(lMax+1)*gnsize + 2*(lMax+1)*(lMax+1)*n  + l*2*(lMax+1) + 2*m + 1]
                                            *Cts[2*t2*(lMax+1)*(lMax+1)*gnsize + 2*(lMax+1)*(lMax+1)*nd + l*2*(lMax+1) + 2*m + 1]);
                            }
                        }
                        nshift++;
                    }
                }
            }
            tshift++;
        }
    }
}
void accumP(double* Phs, double* Ps, int Nt, int lMax, int gnsize, double rCut2, int Ihpos, bool crossover)
{
    int tshift=0;
    int NN = ((gnsize+1)*gnsize)/2;
    int nTypeComb = crossover ? ((Nt+1)*Nt)/2 : Nt;
    for (int t1 = 0; t1 < Nt; t1++) {
        int t2Limit = crossover ? Nt : t1+1;
        for (int t2 = t1; t2 < t2Limit; t2++) {
            for (int l = 0; l < lMax+1; l++) {
                int nshift=0;

                // The power spectrum is multiplied by an l-dependent prefactor that comes
                // from the normalization of the Wigner D matrices. This prefactor is
                // mentioned in the errata of the original SOAP paper: On representing
                // chemical environments, Phys. Rev. B 87, 184115 (2013). Here the square
                // root of the prefactor in the dot-product kernel is used, so that after a
                // possible dot-product the full prefactor is recovered.
                double prefactor = PI*sqrt(8.0/(2.0*l+1.0));

                for (int n = 0; n < gnsize; n++) {
                    for (int nd = n; nd < gnsize; nd++) {
                        Phs[Ihpos*nTypeComb*(lMax+1)*NN + tshift*(lMax+1)*NN + l*NN + nshift] = prefactor*39.478417604*rCut2*Ps[tshift*(lMax+1)*NN + l*NN + nshift];// 16*9.869604401089358*Ps[tshift*(lMax+1)*NN + l*NN + nshift];
                        nshift++;
                    }
                }
            }
            tshift++;
        }
    }
}
void soapGeneral(py::array_t<double> cArr, py::array_t<double> positions, py::array_t<double> HposArr, py::array_t<int> atomicNumbersArr, double rCut, double cutoffPadding, int totalAN, int Nt, int nMax, int lMax, int Hs, double alpha, py::array_t<double> rwArr, py::array_t<double> gssArr, bool crossover)
{
    auto atomicNumbers = atomicNumbersArr.unchecked<1>();
    double *c = (double*)cArr.request().ptr;
    double *Hpos = (double*)HposArr.request().ptr;
    double *rw = (double*)rwArr.request().ptr;
    double *gss = (double*)gssArr.request().ptr;
    double* cf = factorListSet();
    int* isCenter = (int*)malloc( sizeof(int) );
    isCenter[0] = 0;
    const int rsize = 100; // The number of points in the radial integration grid
    double rCut2 = rCut*rCut;
    double* dx = tot;
    double* dy = tot;
    double* dz = tot;
    double* ris = tot;
    double* oOri = tot;
    double* ws  = getws();
    double* oOr = getoOr(rw, rsize);
    double* rw2 = getrw2(rw, rsize);
    double* oO4arri = totrs;
    double* minExp = totrs;
    double* pluExp = totrs;
    int Asize = 0;
    double* Cs = (double*) malloc(2*sd*(lMax+1)*(lMax+1)*nMax);
    double* Cts = (double*) malloc(2*sd*(lMax+1)*(lMax+1)*nMax*Nt);
    double* Ps = crossover ? (double*) malloc((Nt*(Nt+1))/2*sd*(lMax+1)*((nMax+1)*nMax)/2) : (double*) malloc(Nt*sd*(lMax+1)*((nMax+1)*nMax)/2);
    int n_neighbours;

    // Create a mapping between an atomic index and its internal index in the
    // output
    map<int, int> ZIndexMap;
    set<int> atomicNumberSet;
    for (int i = 0; i < totalAN; ++i) {
        atomicNumberSet.insert(atomicNumbers(i));
    };
    int i = 0;
    for (auto it=atomicNumberSet.begin(); it!=atomicNumberSet.end(); ++it) {
        ZIndexMap[*it] = i;
        ++i;
    };

    // Initialize binning
    CellList cellList(positions, rCut+cutoffPadding);

    // Loop through central points
    for (int i = 0; i < Hs; i++) {

        // Get all neighbours for the central atom i
        double ix = Hpos[3*i];
        double iy = Hpos[3*i+1];
        double iz = Hpos[3*i+2];
        CellListResult result = cellList.getNeighboursForPosition(ix, iy, iz);

        // Sort the neighbours by type
        map<int, vector<int>> atomicTypeMap;
        for (const int &idx : result.indices) {
            int Z = atomicNumbers(idx);
            atomicTypeMap[Z].push_back(idx);
        };

        // Loop through neighbours sorted by type
        for (const auto &ZIndexPair : atomicTypeMap) {

            // j is the internal index for this atomic number
            int j = ZIndexMap[ZIndexPair.first];
            int n_neighbours = ZIndexPair.second.size();

            double* Ylmi; double* Flir; double* summed;
            isCenter[0] = 0;

            // Notice that due to the numerical integration the the getDeltas
            // function here has special functionality for positions that are
            // centered on an atom.
//            n_neighbours = getDeltas(dx, dy, dz, ris, rw, rCut, oOri, oO4arri, minExp, pluExp, isCenter, alpha, positions, ix, iy, iz, ZIndexPair.second, rsize, i, j);

            getC(Cs, ws, rw2, gss, summed, rCut, lMax, rsize, nMax, isCenter, alpha);
            accumC(Cts, Cs, lMax, nMax, j);

            free(Flir);
            free(Ylmi);
            free(summed);
        }
        getPs(Ps, Cts,  Nt, lMax, nMax, crossover);
        accumP(c, Ps, Nt, lMax, nMax, rCut2, i, crossover);
    }

    free(cf);
    free(dx);
    free(dy);
    free(dz);
    free(ris);
    free(oOri);
    free(ws);
    free(oOr);
    free(rw2) ;
    free(oO4arri);
    free(minExp);
    free(pluExp);
    free(Cs);
    free(Cts);
    free(Ps);
}
