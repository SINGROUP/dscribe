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
#ifndef SOAPGTO_H
#define SOAPGTO_H

#include <vector>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;

inline int getCrosNum(int n);
inline void getReIm2(double* x, double* y, double* c3, int Asize);
inline void getReIm3(double* x, double* y, double* c2, double* c3, int Asize);
inline void getMulReIm(double* c1, double* c2, double* c3, int Asize);
inline void getMulDouble(double* c1, double* c3, int Asize);
inline int getDeltas(double* x, double* y, double* z, double *positions, double r[3], const vector<int> &indices);
inline void getRsZs(double* x, double* y, double* z,double* r2,double* r4,double* r6,double* r8,double* z2,double* z4,double* z6,double* z8, int size);
void getAlphaBeta(double* aOa, double* bOa, double* alphas, double* betas, int Ns,int lMax, double oOeta, double oOeta3O2);
void getCfactors(double* preCoef, int Asize, double* x, double* y, double* z, double* z2, double* z4, double* z6, double* z8, double* r2, double* r4, double* r6, double* r8, double* ReIm2, double* ReIm3, double* ReIm4, double* ReIm5, double* ReIm6, double* ReIm7, double* ReIm8, double* ReIm9,int totalAN, int lMax, int t2, int t3, int t4, int t5, int t6, int t7, int t8, int t9, int t10, int t11, int t12, int t13, int t14, int t15, int t16, int t17, int t18, int t19, int t20, int t21, int t22, int t23, int t24, int t25, int t26, int t27, int t28, int t29, int t30, int t31, int t32, int t33, int t34, int t35, int t36, int t37, int t38, int t39, int t40, int t41, int t42, int t43, int t44, int t45, int t46, int t47, int t48, int t49, int t50, int t51, int t52, int t53, int t54, int t55, int t56, int t57, int t58, int t59, int t60, int t61, int t62, int t63, int t64, int t65, int t66, int t67, int t68, int t69, int t70, int t71, int t72, int t73, int t74, int t75, int t76, int t77, int t78, int t79, int t80, int t81, int t82, int t83, int t84, int t85, int t86, int t87, int t88, int t89, int t90, int t91, int t92, int t93, int t94, int t95, int t96, int t97, int t98, int t99);
void getC(double* C, double* preCoef, double* x, double* y, double* z,double* r2, double* bOa, double* aOa, double* exes,  int totalAN, int Asize, int Ns, int Ntypes, int lMax, int posI, int typeJ, int Nx2, int Nx3, int Nx4, int Nx5, int Nx6, int Nx7, int Nx8, int Nx9, int Nx10, int Nx11, int Nx12, int Nx13, int Nx14, int Nx15, int Nx16, int Nx17, int Nx18, int Nx19, int Nx20, int Nx21, int Nx22, int Nx23, int Nx24, int Nx25, int Nx26, int Nx27, int Nx28, int Nx29, int Nx30, int Nx31, int Nx32, int Nx33, int Nx34, int Nx35, int Nx36, int Nx37, int Nx38, int Nx39, int Nx40, int Nx41, int Nx42, int Nx43, int Nx44, int Nx45, int Nx46, int Nx47, int Nx48, int Nx49, int Nx50, int Nx51, int Nx52, int Nx53, int Nx54, int Nx55, int Nx56, int Nx57, int Nx58, int Nx59, int Nx60, int Nx61, int Nx62, int Nx63, int Nx64, int Nx65, int Nx66, int Nx67, int Nx68, int Nx69, int Nx70, int Nx71, int Nx72, int Nx73, int Nx74, int Nx75, int Nx76, int Nx77, int Nx78, int Nx79, int Nx80, int Nx81, int Nx82, int Nx83, int Nx84, int Nx85, int Nx86, int Nx87, int Nx88, int Nx89, int Nx90, int Nx91, int Nx92, int Nx93, int Nx94, int Nx95, int Nx96, int Nx97, int Nx98, int Nx99, int t2, int t3, int t4, int t5, int t6, int t7, int t8, int t9, int t10, int t11, int t12, int t13, int t14, int t15, int t16, int t17, int t18, int t19, int t20, int t21, int t22, int t23, int t24, int t25, int t26, int t27, int t28, int t29, int t30, int t31, int t32, int t33, int t34, int t35, int t36, int t37, int t38, int t39, int t40, int t41, int t42, int t43, int t44, int t45, int t46, int t47, int t48, int t49, int t50, int t51, int t52, int t53, int t54, int t55, int t56, int t57, int t58, int t59, int t60, int t61, int t62, int t63, int t64, int t65, int t66, int t67, int t68, int t69, int t70, int t71, int t72, int t73, int t74, int t75, int t76, int t77, int t78, int t79, int t80, int t81, int t82, int t83, int t84, int t85, int t86, int t87, int t88, int t89, int t90, int t91, int t92, int t93, int t94, int t95, int t96, int t97, int t98, int t99);
void getP(double* soapMat, double* Cnnd, int Ns, int Ts, int Hs, int lMax);
void soapGTO(py::array_t<double> c, py::array_t<double> Apos, py::array_t<double> Hpos, py::array_t<double> alphas, py::array_t<double> betas, py::array_t<int> atomicNumbers, double rCut, double cutoffPadding, int totalAN, int Nt, int Ns, int lMax, int Hs, double eta, bool crossover);

#endif
