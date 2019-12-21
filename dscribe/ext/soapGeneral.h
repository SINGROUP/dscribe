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
int getFilteredPos(double* x, double* y, double* z,double* xNow, double* yNow, double* zNow, double* ri, double* rw, double rCut, double* oOri, double* oO4arri, double* minExp, double* pluExp,int* isCenter, double alpha, double* Apos, double* Hpos,int* typeNs, int rsize, int Ihpos, int Itype);
double* getFlir(double* oO4arri,double* ri, double* minExp, double* pluExp, int icount, int rsize, int lMax);
double legendre_poly(int l, int m, double x);
double* getYlmi(double* x, double* y, double* z, double* oOri, double* cf, int icount, int lMax);
double* getIntegrand(double* Flir, double* Ylmi,int rsize, int icount, int lMax);
void getC(double* Cs, double* ws, double* rw2, double * gns, double* summed, double rCut,int lMax, int rsize, int gnsize,int* isCenter, double alpha);
void accumC(double* Cts, double* Cs, int lMax, int gnsize, int typeI);
void getPs(double* Ps, double* Cts,  int Nt, int lMax, int gnsize);
void accumP(double* Phs, double* Ps, int Nt, int lMax, int gnsize, double rCut2, int Ihpos);
double* soapGeneral(py::array_t<double, py::array::c_style | py::array::forcecast> cArr, py::array_t<double, py::array::c_style | py::array::forcecast> AposArr, py::array_t<double, py::array::c_style | py::array::forcecast> HposArr, py::array_t<int, py::array::c_style | py::array::forcecast> typeNsArr, double rCut, int totalAN,int Nt,int gnsize, int lMax, int Hs, double alpha, py::array_t<double, py::array::c_style | py::array::forcecast> rwArr, py::array_t<double, py::array::c_style | py::array::forcecast> gssArr);

#endif
