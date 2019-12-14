#ifndef SOAPGENERAL_H
#define SOAPGENERAL_H

#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;

double* factorListSet();
double* getws();
double factorY(int l, int m, double* c);
double* getoOr(double* r, int rsize);
double* getrw2(double* r, int rsize);
void expMs(double* rExpDiff, double alpha, double* r, double* ri, int isize, int rsize);
void expPs(double* rExpSum, double alpha, double* r, double* ri, int isize, int rsize);
void getDeltas(double* xNow, double* yNow, double* zNow, double* ri, double* rw, double rCut, double* oOri, double* oO4arri, double* minExp, double* pluExp,int* isCenter, double alpha, const py::array_t<double> &positions, const double ix, const double iy, const double iz, const vector<int> &indices, int rsize, int Ihpos, int Itype);
double* getFlir(double* oO4arri,double* ri, double* minExp, double* pluExp, int icount, int rsize, int lMax);
double legendre_poly(int l, int m, double x);
double* getYlmi(double* x, double* y, double* z, double* oOri, double* cf, int icount, int lMax);
double* getIntegrand(double* Flir, double* Ylmi,int rsize, int icount, int lMax);
void getC(double* Cs, double* ws, double* rw2, double * gns, double* summed, double rCut,int lMax, int rsize, int gnsize,int* isCenter, double alpha);
void accumC(double* Cts, double* Cs, int lMax, int gnsize, int typeI);
void getPs(double* Ps, double* Cts,  int Nt, int lMax, int gnsize, bool crossover);
void accumP(double* Phs, double* Ps, int Nt, int lMax, int gnsize, double rCut2, int Ihpos, bool crossover);
void soapGeneral(py::array_t<double> cArr, py::array_t<double> AposArr, py::array_t<double> HposArr, py::array_t<int> atomicNumbersArr, double rCut, double cutoffPadding, int totalAN,int Nt,int nmax, int lMax, int Hs, double alpha, py::array_t<double> rwArr, py::array_t<double> gssArr, bool crossover);

#endif
