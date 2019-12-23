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
#include <stdlib.h>
#include <time.h>
#include <string>
#include <map>
#include <set>
#include "soapGTO.h"
#include "celllist.h"

#define PI2 9.86960440108936
#define PI 3.14159265359
#define PIHalf 1.57079632679490
//===========================================================
inline int getCrosNum(int n)
{
  return n*(n+1)/2;
}
//===========================================================
inline void getReIm2(double* x, double* y, double* c3, int Asize){
  for(int i = 0; i < Asize; i++){
    c3[2*i  ] = x[i]*x[i]-y[i]*y[i];
    c3[2*i+1] = 2*y[i]*x[i];
  }
}
//===========================================================
inline void getReIm3(double* x, double* y, double* c2, double* c3, int Asize){
  for(int i = 0; i < Asize; i++){
    c3[2*i  ] = x[i]*c2[2*i] - y[i]*c2[2*i + 1];
    c3[2*i+1] = x[i]*c2[2*i+1] + y[i]*c2[2*i  ];
  }
}
//===========================================================
inline void getMulReIm(double* c1, double* c2, double* c3, int Asize){
  for(int i = 0; i < Asize; i++){
    c3[2*i  ] = c1[2*i  ]*c2[2*i  ] - c1[2*i+1]*c2[2*i+1];
    c3[2*i+1] = c1[2*i  ]*c2[2*i+1] + c1[2*i+1]*c2[2*i  ];
  }
}
//===========================================================
inline void getMulDouble(double* c1, double* c3, int Asize){
  for(int i = 0; i < Asize; i++){
    c3[2*i  ] = c1[2*i]*c1[2*i] - c1[2*i+1]*c1[2*i+1];
    c3[2*i+1] = 2*c1[2*i]*c1[2*i+1];
  }
}
//================================================================
inline void getDeltas(double* x, double* y, double* z, const py::array_t<double> &positions, const double ix, const double iy, const double iz, const vector<int> &indices){

    int count = 0;
    auto pos = positions.unchecked<2>();
    for (const int &idx : indices) {
        x[count] = pos(idx, 0) - ix;
        y[count] = pos(idx, 1) - iy;
        z[count] = pos(idx, 2) - iz;
        count++;
    };
}

//================================================================
inline void getRsZs(double* x, double* y, double* z,double* r2,double* r4,double* r6,double* r8,double* z2,double* z4,double* z6,double* z8, int size){
  for(int i = 0; i < size; i++){
    r2[i] = x[i]*x[i] + y[i]*y[i] + z[i]*z[i];
    r4[i] = r2[i]*r2[i]; r6[i] = r2[i]*r4[i]; r8[i] = r4[i]*r4[i];
    z2[i] = z[i]*z[i]; z4[i] = z2[i]*z2[i]; z6[i] = z2[i]*z4[i]; z8[i] = z4[i]*z4[i];
  }
}
//================================================================
void getAlphaBeta(double* aOa, double* bOa, double* alphas, double* betas, int Ns,int lMax, double oOeta, double oOeta3O2){

  int  NsNs = Ns*Ns;
  double  oneO1alpha;  double  oneO1alpha2; double  oneO1alpha3;
  double  oneO1alpha4; double  oneO1alpha5; double  oneO1alpha6;
  double  oneO1alpha7; double  oneO1alpha8; double  oneO1alpha9;
  double  oneO1alpha10;
  double  oneO1alphaSqrt;// = (double*) malloc(Ns*sizeof(double));
  double  oneO1alphaSqrtX;

  for(int k = 0; k < Ns; k++){
    oneO1alpha = 1.0/(1.0 + oOeta*alphas[k]);
    oneO1alphaSqrt = sqrt(oneO1alpha);
    aOa[k] = -alphas[k]*oneO1alpha; //got alpha_0k
    oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha;
    for(int n = 0; n < Ns; n++){ bOa[n*Ns + k] = oOeta3O2*betas[n*Ns + k]*oneO1alphaSqrtX;} // got beta_0nk
  }
  if(lMax > 0){
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[Ns + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[Ns + k] = -alphas[Ns + k]*oneO1alpha; //got alpha_1k
      oneO1alpha2 = oneO1alpha*oneO1alpha; oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha2;
      for(int n = 0; n < Ns; n++){ bOa[NsNs + n*Ns + k] = oOeta3O2*betas[NsNs + n*Ns + k]*oneO1alphaSqrtX;} // got beta_1nk
    }
  } if(lMax > 1){
    int shift1 = 2*Ns; int shift2 = 2*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_2k
      oneO1alpha3 = oneO1alpha*oneO1alpha*oneO1alpha; oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha3;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;} // got beta_2nk
    }
  } if(lMax > 2){
    int shift1 = 3*Ns; int shift2 = 3*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_3k
      oneO1alpha4 = oneO1alpha*oneO1alpha*oneO1alpha*oneO1alpha; oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha4;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;} // got beta_3nk
    }
  } if(lMax > 3){
    int shift1 = 4*Ns; int shift2 = 4*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_4k
      oneO1alpha5 = pow(oneO1alpha,5); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha5;
      for(int n = 0; n < Ns; n++){ bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;} // got beta_4nk
    }
  } if(lMax > 4){
    int shift1 = 5*Ns; int shift2 = 5*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_5k
      oneO1alpha6 = pow(oneO1alpha,6); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha6;
      for(int n = 0; n < Ns; n++){ bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;} // got beta_5nk
    }
  } if(lMax > 5){
    int shift1 = 6*Ns; int shift2 = 6*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_6k
      oneO1alpha7 = pow(oneO1alpha,7); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha7;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;} // got beta_6nk
    }
  } if(lMax > 6){
    int shift1 = 7*Ns; int shift2 = 7*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_7k
      oneO1alpha8 = pow(oneO1alpha,8); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha8;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;} // got beta_7nk
    }
  } if(lMax > 7){
    int shift1 = 8*Ns; int shift2 = 8*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_8k
      oneO1alpha9 = pow(oneO1alpha,9); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha9;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;} // got beta_8nk
    }
  }if(lMax > 8){
    int shift1 = 9*Ns; int shift2 = 9*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_9k
      oneO1alpha10 = pow(oneO1alpha,10); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha10;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;} // got beta_9nk
    }
  }
}
//================================================================
void getCfactors(double* preCoef, int Asize, double* x, double* y, double* z, double* z2, double* z4, double* z6, double* z8, double* r2, double* r4, double* r6, double* r8, double* ReIm2, double* ReIm3, double* ReIm4, double* ReIm5, double* ReIm6, double* ReIm7, double* ReIm8, double* ReIm9,int totalAN, int lMax, int t2, int t3, int t4, int t5, int t6, int t7, int t8, int t9, int t10, int t11, int t12, int t13, int t14, int t15, int t16, int t17, int t18, int t19, int t20, int t21, int t22, int t23, int t24, int t25, int t26, int t27, int t28, int t29, int t30, int t31, int t32, int t33, int t34, int t35, int t36, int t37, int t38, int t39, int t40, int t41, int t42, int t43, int t44, int t45, int t46, int t47, int t48, int t49, int t50, int t51, int t52, int t53, int t54, int t55, int t56, int t57, int t58, int t59, int t60, int t61, int t62, int t63, int t64, int t65, int t66, int t67, int t68, int t69, int t70, int t71, int t72, int t73, int t74, int t75, int t76, int t77, int t78, int t79, int t80, int t81, int t82, int t83, int t84, int t85, int t86, int t87, int t88, int t89, int t90, int t91, int t92, int t93, int t94, int t95, int t96, int t97, int t98, int t99){
  double c20c;double c30c;double c31c;double c40c;double c41c;double c42c;
  double c50c;double c51c;double c52c;double c53c;double c60c;double c61c;
  double c62c;double c63c;double c64c;double c70c;double c71c;double c72c;
  double c73c;double c74c;double c75c;double c80c;double c81c;double c82c;
  double c83c;double c84c;double c85c;double c86c;double c90c;double c91c;
  double c92c;double c93c;double c94c;double c95c;double c96c;double c97c;

  getReIm2(x, y, ReIm2, Asize);
  getReIm3(x, y, ReIm2, ReIm3, Asize);
  getMulDouble(ReIm2, ReIm4, Asize);
  getMulReIm(ReIm2, ReIm3, ReIm5, Asize);
  getMulDouble(ReIm3, ReIm6, Asize);
  getMulReIm(ReIm3, ReIm4, ReIm7, Asize);
  getMulDouble(ReIm4, ReIm8, Asize);
  getMulReIm(ReIm4, ReIm5, ReIm9, Asize);
  int i2;

  for (int i = 0; i < Asize; i++) {
    i2 = 2*i;
    c20c=3*z2[i]-r2[i];
    if (lMax > 2){
      c30c=5*z2[i]-3*r2[i];
      c31c=5*z2[i]-r2[i];
    }
    if (lMax > 3){
      c40c=35*z4[i]-30*z2[i]*r2[i]+3*r4[i];
      c41c=7*z2[i]-3*r2[i];
      c42c=7*z2[i]-r2[i];
    }
    if (lMax > 4){
      c50c=63*z4[i]-70*z2[i]*r2[i]+15*r4[i];
      c51c=21*z4[i]-14*z2[i]*r2[i]+r4[i];
      c52c=3*z2[i]-r2[i];
      c53c=9*z2[i]-r2[i];
    }
    if (lMax > 5){
      c60c=231*z6[i] - 315*z4[i]*r2[i] + 105*z2[i]*r4[i] - 5*r6[i];
      c61c=33*z4[i] - 30*z2[i]*r2[i] + 5*r4[i];
      c62c=33*z4[i] - 18*z2[i]*r2[i] + r4[i];
      c63c=11*z2[i] - 3*r2[i];
      c64c=11*z2[i] - r2[i];
    }
    if (lMax > 6){
      c70c=429*z6[i]-693*z4[i]*r2[i]+315*z2[i]*r4[i]-35*r6[i];
      c71c=429*z6[i]-495*z4[i]*r2[i]+135*z2[i]*r4[i]-5*r6[i];
      c72c=143*z4[i]-110*z2[i]*r2[i]+15*r4[i];
      c73c=143*z4[i]-66*z2[i]*r2[i]+3*r4[i];
      c74c=13*z2[i]-3*r2[i];
      c75c=13*z2[i]-r2[i];
    }
    if (lMax > 7){
      c80c=6435*z8[i]-12012*z6[i]*r2[i]+6930*z4[i]*r4[i]-1260*z2[i]*r6[i]+35*r8[i];
      c81c=715*z6[i]-1001*z4[i]*r2[i]+385*z2[i]*r4[i]-35*r6[i];
      c82c=143*z6[i]-143*z4[i]*r2[i]+33*z2[i]*r4[i]-r6[i];
      c83c=39*z4[i]-26*z2[i]*r2[i]+3*r4[i];
      c84c=65*z4[i]-26*z2[i]*r2[i]+r4[i];
      c85c=5*z2[i]-r2[i];
      c86c=15*z2[i]-r2[i];
    }
    if (lMax > 8){
      c90c=12155*z8[i]-25740*z6[i]*r2[i]+18018*z4[i]*r4[i] -4620*z2[i]*r6[i]+315*r8[i];
      c91c=2431*z8[i]-4004*z6[i]*r2[i]+2002*z4[i]*r4[i]-308*z2[i]*r6[i] + 7*r8[i];
      c92c=221*z6[i]-273*z4[i]*r2[i]+91*z2[i]*r4[i]-7*r6[i];
      c93c=221*z6[i]-195*z4[i]*r2[i]+39*z2[i]*r4[i]-r6[i];
      c94c=17*z4[i]-10*z2[i]*r2[i]+r4[i];
      c95c=85*z4[i]-30*z2[i]*r2[i]+r4[i];
      c96c=17*z2[i]-3*r2[i];
      c97c=17*z2[i]-r2[i];
    }
    /*c20  */  preCoef[        +i] = c20c;
    /*c21Re*/  preCoef[totalAN+i] = z[i]*x[i];
    /*c21Im*/  preCoef[t2+i] = z[i]*y[i];
    /*c22Re*/  preCoef[t3+i] =      ReIm2[2*i];
    /*c22Im*/  preCoef[t4+i] =      ReIm2[i2+1];
    if (lMax > 2){
      /*c30  */  preCoef[t5+i] = c30c*z[i];
      /*c31Re*/  preCoef[t6+i] =       x[i]*c31c;
      /*c31Im*/  preCoef[t7+i] =       y[i]*c31c;
      /*c32Re*/  preCoef[t8+i] = z[i]*ReIm2[i2];
      /*c32Im*/  preCoef[t9+i] = z[i]*ReIm2[i2+1];
      /*c33Re*/  preCoef[t10+i] =      ReIm3[i2  ];
      /*c33Im*/  preCoef[t11+i] =      ReIm3[i2+1];
    }
    if (lMax > 3){
      /*c40  */  preCoef[t12+i] = c40c;
      /*c41Re*/  preCoef[t13+i] = z[i]*x[i]*c41c;
      /*c41Im*/  preCoef[t14+i] = z[i]*y[i]*c41c;
      /*c42Re*/  preCoef[t15+i] =      ReIm2[i2  ]*c42c;
      /*c42Im*/  preCoef[t16+i] =      ReIm2[i2+1]*c42c;
      /*c43Re*/  preCoef[t17+i] = z[i]*ReIm3[i2  ];
      /*c43Im*/  preCoef[t18+i] = z[i]*ReIm3[i2+1];
      /*c44Re*/  preCoef[t19+i] =      ReIm4[i2  ];
      /*c44Im*/  preCoef[t20+i] =      ReIm4[i2+1];
    }
    if (lMax > 4){
      /*c50  */  preCoef[t21+i] = c50c*z[i];
      /*c51Re*/  preCoef[t22+i] =      x[i]*c51c;
      /*c51Im*/  preCoef[t23+i] =      y[i]*c51c;
      /*c52Re*/  preCoef[t24+i] = z[i]*ReIm2[i2  ]*c52c;
      /*c52Im*/  preCoef[t25+i] = z[i]*ReIm2[i2+1]*c52c;
      /*c53Re*/  preCoef[t26+i] =      ReIm3[i2  ]*c53c;
      /*c53Im*/  preCoef[t27+i] =      ReIm3[i2+1]*c53c;
      /*c54Re*/  preCoef[t28+i] = z[i]*ReIm4[i2  ];
      /*c54Im*/  preCoef[t29+i] = z[i]*ReIm4[i2+1];
      /*c55Re*/  preCoef[t30+i] =      ReIm5[i2  ];
      /*c55Im*/  preCoef[t31+i] =      ReIm5[i2+1];
    }
    if (lMax > 5){
      /*c60  */  preCoef[t32+i] = c60c;
      /*c61Re*/  preCoef[t33+i] = z[i]*x[i]*c61c;
      /*c61Im*/  preCoef[t34+i] = z[i]*y[i]*c61c;
      /*c62Re*/  preCoef[t35+i] =      ReIm2[i2  ]*c62c;
      /*c62Im*/  preCoef[t36+i] =      ReIm2[i2+1]*c62c;
      /*c63Re*/  preCoef[t37+i] = z[i]*ReIm3[i2  ]*c63c;
      /*c63Im*/  preCoef[t38+i] = z[i]*ReIm3[i2+1]*c63c;
      /*c64Re*/  preCoef[t39+i] =      ReIm4[i2  ]*c64c;
      /*c64Im*/  preCoef[t40+i] =      ReIm4[i2+1]*c64c;
      /*c65Re*/  preCoef[t41+i] = z[i]*ReIm5[i2  ];
      /*c65Im*/  preCoef[t42+i] = z[i]*ReIm5[i2+1];
      /*c66Re*/  preCoef[t43+i] =      ReIm6[i2  ];
      /*c66Im*/  preCoef[t44+i] =      ReIm6[i2+1];
    }
    if (lMax > 6){
      /*c70  */  preCoef[t45+i] = c70c*z[i];
      /*c71Re*/  preCoef[t46+i] = x[i]*c71c;
      /*c71Im*/  preCoef[t47+i] = y[i]*c71c;
      /*c72Re*/  preCoef[t48+i] = z[i]*ReIm2[i2  ]*c72c;
      /*c72Im*/  preCoef[t49+i] = z[i]*ReIm2[i2+1]*c72c;
      /*c73Re*/  preCoef[t50+i] =      ReIm3[i2  ]*c73c;
      /*c73Im*/  preCoef[t51+i] =      ReIm3[i2+1]*c73c;
      /*c74Re*/  preCoef[t52+i] = z[i]*ReIm4[i2  ]*c74c;
      /*c74Im*/  preCoef[t53+i] = z[i]*ReIm4[i2+1]*c74c;
      /*c75Re*/  preCoef[t54+i] =      ReIm5[i2  ]*c75c;
      /*c75Im*/  preCoef[t55+i] =      ReIm5[i2+1]*c75c;
      /*c76Re*/  preCoef[t56+i] = z[i]*ReIm6[i2  ];
      /*c76Im*/  preCoef[t57+i] = z[i]*ReIm6[i2+1];
      /*c77Re*/  preCoef[t58+i] =      ReIm7[i2  ];
      /*c77Im*/  preCoef[t59+i] =      ReIm7[i2+1];
    }
    if (lMax > 7){
      /*c80  */  preCoef[t60+i] = c80c;
      /*c81Re*/  preCoef[t61+i] = z[i]*x[i]*c81c;
      /*c81Im*/  preCoef[t62+i] = z[i]*y[i]*c81c;
      /*c82Re*/  preCoef[t63+i] =      ReIm2[i2  ]*c82c;
      /*c82Im*/  preCoef[t64+i] =      ReIm2[i2+1]*c82c;
      /*c83Re*/  preCoef[t65+i] = z[i]*ReIm3[i2  ]*c83c;
      /*c83Im*/  preCoef[t66+i] = z[i]*ReIm3[i2+1]*c83c;
      /*c84Re*/  preCoef[t67+i] =      ReIm4[i2  ]*c84c;
      /*c84Im*/  preCoef[t68+i] =      ReIm4[i2+1]*c84c;
      /*c85Re*/  preCoef[t69+i] = z[i]*ReIm5[i2  ]*c85c;
      /*c85Im*/  preCoef[t70+i] = z[i]*ReIm5[i2+1]*c85c;
      /*c86Re*/  preCoef[t71+i] =      ReIm6[i2  ]*c86c;
      /*c86Im*/  preCoef[t72+i] =      ReIm6[i2+1]*c86c;
      /*c87Re*/  preCoef[t73+i] = z[i]*ReIm7[i2  ];
      /*c87Im*/  preCoef[t74+i] = z[i]*ReIm7[i2+1];
      /*c88Re*/  preCoef[t75+i] =      ReIm8[i2  ];
      /*c88Im*/  preCoef[t76+i] =      ReIm8[i2+1];
    }
    if (lMax > 8){
      /*c90  */  preCoef[t77+i] = c90c*z[i];
      /*c91Re*/  preCoef[t78+i] = x[i]*c91c;
      /*c91Im*/  preCoef[t79+i] = y[i]*c91c;
      /*c92Re*/  preCoef[t80+i] = z[i]*ReIm2[i2  ]*c92c;
      /*c92Im*/  preCoef[t81+i] = z[i]*ReIm2[i2+1]*c92c;
      /*c93Re*/  preCoef[t82+i] =      ReIm3[i2  ]*c93c;
      /*c93Im*/  preCoef[t83+i] =      ReIm3[i2+1]*c93c;
      /*c94Re*/  preCoef[t84+i] = z[i]*ReIm4[i2  ]*c94c;
      /*c94Im*/  preCoef[t85+i] = z[i]*ReIm4[i2+1]*c94c;
      /*c95Re*/  preCoef[t86+i] =      ReIm5[i2  ]*c95c;
      /*c95Im*/  preCoef[t87+i] =      ReIm5[i2+1]*c95c;
      /*c96Re*/  preCoef[t88+i] = z[i]*ReIm6[i2  ]*c96c;
      /*c96Im*/  preCoef[t89+i] = z[i]*ReIm6[i2+1]*c96c;
      /*c97Re*/  preCoef[t90+i] =      ReIm7[i2  ]*c97c;
      /*c97Im*/  preCoef[t91+i] =      ReIm7[i2+1]*c97c;
      /*c98Re*/  preCoef[t92+i] = z[i]*ReIm8[i2  ];
      /*c98Im*/  preCoef[t93+i] = z[i]*ReIm8[i2+1];
      /*c99Re*/  preCoef[t94+i] =      ReIm9[i2  ];
      /*c99Im*/  preCoef[t95+i] =      ReIm9[i2+1];
    }
  }
}
//================================================================
void getC(double* C, double* preCoef, double* x, double* y, double* z,double* r2, double* bOa, double* aOa, double* exes,  int totalAN, int Asize, int Ns, int Ntypes, int lMax, int posI, int typeJ, int Nx2, int Nx3, int Nx4, int Nx5, int Nx6, int Nx7, int Nx8, int Nx9, int Nx10, int Nx11, int Nx12, int Nx13, int Nx14, int Nx15, int Nx16, int Nx17, int Nx18, int Nx19, int Nx20, int Nx21, int Nx22, int Nx23, int Nx24, int Nx25, int Nx26, int Nx27, int Nx28, int Nx29, int Nx30, int Nx31, int Nx32, int Nx33, int Nx34, int Nx35, int Nx36, int Nx37, int Nx38, int Nx39, int Nx40, int Nx41, int Nx42, int Nx43, int Nx44, int Nx45, int Nx46, int Nx47, int Nx48, int Nx49, int Nx50, int Nx51, int Nx52, int Nx53, int Nx54, int Nx55, int Nx56, int Nx57, int Nx58, int Nx59, int Nx60, int Nx61, int Nx62, int Nx63, int Nx64, int Nx65, int Nx66, int Nx67, int Nx68, int Nx69, int Nx70, int Nx71, int Nx72, int Nx73, int Nx74, int Nx75, int Nx76, int Nx77, int Nx78, int Nx79, int Nx80, int Nx81, int Nx82, int Nx83, int Nx84, int Nx85, int Nx86, int Nx87, int Nx88, int Nx89, int Nx90, int Nx91, int Nx92, int Nx93, int Nx94, int Nx95, int Nx96, int Nx97, int Nx98, int Nx99, int t2, int t3, int t4, int t5, int t6, int t7, int t8, int t9, int t10, int t11, int t12, int t13, int t14, int t15, int t16, int t17, int t18, int t19, int t20, int t21, int t22, int t23, int t24, int t25, int t26, int t27, int t28, int t29, int t30, int t31, int t32, int t33, int t34, int t35, int t36, int t37, int t38, int t39, int t40, int t41, int t42, int t43, int t44, int t45, int t46, int t47, int t48, int t49, int t50, int t51, int t52, int t53, int t54, int t55, int t56, int t57, int t58, int t59, int t60, int t61, int t62, int t63, int t64, int t65, int t66, int t67, int t68, int t69, int t70, int t71, int t72, int t73, int t74, int t75, int t76, int t77, int t78, int t79, int t80, int t81, int t82, int t83, int t84, int t85, int t86, int t87, int t88, int t89, int t90, int t91, int t92, int t93, int t94, int t95, int t96, int t97, int t98, int t99){

  if(Asize == 0){return;}
  double sumMe = 0; int NsNs = Ns*Ns;  int NsJ = 100*Ns*typeJ; int LNsNs;
  int LNs; int NsTsI = 100*Ns*Ntypes*posI;
  for(int k = 0; k < Ns; k++){
    sumMe = 0; for(int i = 0; i < Asize; i++){ sumMe += exp(aOa[k]*r2[i]);}
    for(int n = 0; n < Ns; n++){ C[NsTsI + NsJ + n] += bOa[n*Ns + k]*sumMe; }
  } if(lMax > 0) { LNsNs=NsNs; LNs=Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c10*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*z[i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c11Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*x[i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx2 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c11Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*y[i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx3 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }} if(lMax > 1) { LNsNs=2*NsNs; LNs=2*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c20*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*preCoef[i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx4 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c21Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*preCoef[totalAN + i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx5 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c21Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*preCoef[t2+ i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx6 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c22Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*preCoef[t3+ i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx7 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c22Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*preCoef[t4+ i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx8 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
  }} if(lMax > 2) { LNsNs=3*NsNs; LNs=3*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c30*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t5+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx9 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c31Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t6+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx10 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c31Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t7+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx11 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c32Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t8+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx12 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c32Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t9+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx13 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c33Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t10+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx14 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c33Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t11+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx15 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
  }} if(lMax > 3) { LNsNs=4*NsNs; LNs=4*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c40*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t12+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx16 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c41Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t13+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx17 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c41Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t14+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx18 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c42Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t15+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx19 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c42Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t16+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx20 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c43Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t17+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx21 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c43Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t18+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx22 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c44Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t19+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx23 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c44Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t20+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx24 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }

  }} if(lMax > 4) { LNsNs=5*NsNs; LNs=5*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c50*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t21+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx25 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c51Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t22+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx26 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c51Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t23+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx27 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c52Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t24+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx28 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c52Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t25+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx29 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c53Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t26+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx30 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c53Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t27+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx31 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c54Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t28+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx32 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c54Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t29+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx33 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c55Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t30+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx34 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c55Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t31+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx35 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }} if(lMax > 5) { LNsNs=6*NsNs; LNs=6*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c60*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t32+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx36 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c61Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t33+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx37 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c61Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t34+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx38 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c62Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t35+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx39 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c62Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t36+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx40 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c63Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t37+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx41 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c63Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t38+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx42 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c64Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t39+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx43 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c64Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t40+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx44 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c65Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t41+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx45 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c65Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t42+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx46 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c66Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t43+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx47 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c66Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t44+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx48 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }} if(lMax > 6) { LNsNs=7*NsNs; LNs=7*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c70*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t45+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx49 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c71Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t46+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx50 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c71Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t47+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx51 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c72Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t48+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx52 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c72Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t49+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx53 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c73Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t50+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx54 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c73Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t51+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx55 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c74Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t52+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx56 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c74Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t53+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx57 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c75Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t54+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx58 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c75Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t55+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx59 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c76Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t56+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx60 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c76Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t57+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx61 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c77Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t58+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx62 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c77Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t59+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx63 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }} if(lMax > 7) { LNsNs=8*NsNs; LNs=8*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c80*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t60+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx64 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c81Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t61+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx65 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c81Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t62+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx66 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c82Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t63+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx67 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c82Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t64+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx68 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c83Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t65+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx69 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c83Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t66+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx70 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c84Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t67+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx71 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c84Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t68+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx72 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c85Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t69+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx73 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c85Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t70+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx74 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c86Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t71+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx75 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c86Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t72+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx76 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c87Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t73+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx77 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c87Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t74+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx78 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c88Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t75+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx79 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c88Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t76+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx80 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }} if(lMax > 8) { LNsNs=9*NsNs; LNs=9*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c90*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t77+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx81 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c91Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t78+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx82 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c91Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t79+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx83 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c92Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t80+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx84 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c92Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t81+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx85 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c93Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t82+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx86 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c93Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t83+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx87 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c94Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t84+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx88 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c94Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t85+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx89 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c95Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t86+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx90 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c95Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t87+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx91 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c96Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t88+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx92 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c96Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t89+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx93 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c97Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t90+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx94 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c97Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t91+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx95 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c98Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t92+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx96 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c98Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t93+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx97 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c99Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t94+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx98 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c99Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[t95+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Nx99 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }}
}
//=======================================================================
/**
 * Used to calculate the partial power spectrum without crossover.
 */
void getPNoCross(double* soapMat, double* Cnnd, int Ns, int Ts, int Hs, int lMax){
  int NsTs100 = Ns*Ts*100;
  int Ns100 = Ns*100;
  int NsNs = (Ns*(Ns+1))/2;
  int NsNsLmax = NsNs*(lMax+1);
  int NsNsLmaxTs = NsNsLmax*Ts;
  int shiftN = 0;

  //  double   cs0  = pow(PIHalf,2);
  //  double   cs1  = pow(2.7206990464,2);
  //  double cs2  = 2*pow(1.9238247452,2); double   cs3  = pow(1.7562036828,2); double cs4  = 2*pow(4.3018029072,2);
  //  double cs5  = 2*pow(2.1509014536,2); double   cs6  = pow(2.0779682205,2); double cs7  = 2*pow(1.7995732672,2);
  //  double cs8  = 2*pow(5.6907503408,2); double cs9  = 2*pow(2.3232390981,2); double   cs10 = pow(0.5890486225,2);
  //  double cs11 = 2*pow(2.6343055241,2); double cs12 = 2*pow(1.8627352998,2); double cs13 = 2*pow(6.9697172942,2);
  //  double cs14 = 2*pow(2.4641671809,2); double   cs15 = pow(0.6512177548,2); double cs16 = 2*pow(1.7834332706,2);
  //  double cs17 = 2*pow(9.4370418280,2); double cs18 = 2*pow(1.9263280966,2); double cs19 = 2*pow(8.1727179596,2);
  //  double cs20 = 2*pow(2.5844403427,2); double   cs21 = pow(0.3539741687,2); double cs22 = 2*pow(2.2940148014,2);
  //  double cs23 = 2*pow(1.8135779397,2); double cs24 = 2*pow(3.6271558793,2); double cs25 = 2*pow(1.9866750947,2);
  //  double cs26 = 2*pow(9.3183321738,2); double cs27 = 2*pow(2.6899707945,2); double   cs28 = pow(0.3802292509,2);
  //  double cs29 = 2*pow(0.3556718963,2); double cs30 = 2*pow(0.8712146618,2); double cs31 = 2*pow(0.6160417952,2);
  //  double cs32 = 2*pow(4.0863589798,2); double cs33 = 2*pow(2.0431794899,2); double cs34 = 2*pow(10.418212089,2);
  //  double cs35 = 2*pow(2.7843843014,2); double   cs36 = pow(0.0505981185,2); double cs37 = 2*pow(0.4293392727,2);
  //  double cs38 = 2*pow(1.7960550366,2); double cs39 = 2*pow(4.8637400313,2); double cs40 = 2*pow(1.8837184141,2);
  //  double cs41 = 2*pow(13.583686661,2); double cs42 = 2*pow(2.0960083567,2); double cs43 = 2*pow(11.480310577,2);
  //  double cs44 = 2*pow(2.8700776442,2); double   cs45 = pow(0.0534917379,2); double cs46 = 2*pow(0.2537335916,2);
  //  double cs47 = 2*pow(2.3802320735,2); double cs48 = 2*pow(1.8179322747,2); double cs49 = 2*pow(16.055543121,2);
  //  double cs50 = 2*pow(1.9190044477,2); double cs51 = 2*pow(4.9548481782,2); double cs52 = 2*pow(2.1455121971,2);
  //  double cs53 = 2*pow(12.510378411,2); double cs54 = 2*pow(2.9487244699,2);
  double cs0=2.4674011003; double cs1=7.4022033011; double cs2=7.4022033005;
  double cs3=3.0842513755; double cs4=37.0110165048; double cs5=9.2527541262;
  double cs6=4.3179519254; double cs7=6.4769278880; double cs8=64.7692788826;
  double cs9=10.7948798139; double cs10=0.3469782797; double cs11=13.8791311886;
  double cs12=6.9395655942; double cs13=97.1539183221; double cs14=12.1442397908;
  double cs15=0.4240845642; double cs16=6.3612684614; double cs17=178.1155169268;
  double cs18=7.4214798715; double cs19=133.5866376943; double cs20=13.3586637700;
  double cs21=0.1252977121; double cs22=10.5250078181; double cs23=6.5781298867;
  double cs24=26.3125195455; double cs25=7.8937558638; double cs26=173.6626290026;
  double cs27=14.4718857505; double cs28=0.1445742832; double cs29=0.2530049956;
  double cs30=1.5180299739; double cs31=0.7590149869; double cs32=33.3966594236;
  double cs33=8.3491648559; double cs34=217.0782862628; double cs35=15.5055918758;
  double cs36=0.0025601696; double cs37=0.3686644222; double cs38=6.4516273890;
  double cs39=47.3119341841; double cs40=7.0967901272; double cs41=369.0330866085;
  double cs42=8.7865020627; double cs43=263.5950618888; double cs44=16.4746913675;
  double cs45=0.0028613660; double cs46=0.1287614710; double cs47=11.3310094474;
  double cs48=6.6097555108; double cs49=515.5609298206; double cs50=7.3651561406;
  double cs51=49.1010409380; double cs52=9.2064451758; double cs53=313.0191359728;
  double cs54=17.3899519988;

  // The power spectrum is multiplied by an l-dependent prefactor that comes
  // from the normalization of the Wigner D matrices. This prefactor is
  // mentioned in the arrata of the original SOAP paper: On representing
  // chemical environments, Phys. Rev. B 87, 184115 (2013). Here the square
  // root of the prefactor in the dot-product kernel is used, so that after a
  // possible dot-product the full prefactor is recovered.

  // SUM M's UP!
  double prel0 = PI*sqrt(8.0/(1.0));
  for(int i = 0; i < Hs; i++){
    for(int j = 0; j < Ts; j++){
      shiftN = 0;
      for(int k = 0; k < Ns; k++){
        for(int kd = k; kd < Ns; kd++){
          soapMat[NsNsLmaxTs*i+ NsNsLmax*j+ 0 +shiftN] = prel0*(
            cs0*Cnnd[NsTs100*i + Ns100*j + 0 + k]*Cnnd[NsTs100*i + Ns100*j + 0 + kd]);
          shiftN++;
        }
      }
    }
  } if (lMax > 0) {
    double prel1 = PI*sqrt(8.0/(2.0*1.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ NsNs + shiftN] = prel1*(
              cs1*Cnnd[NsTs100*i + Ns100*j + 1*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 1*Ns + kd]
             +cs2*Cnnd[NsTs100*i + Ns100*j + 2*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 2*Ns + kd]
             +cs2*Cnnd[NsTs100*i + Ns100*j + 3*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 3*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if (lMax > 1) {
    double prel2 = PI*sqrt(8.0/(2.0*2.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 2*NsNs + shiftN] = prel2*(
              cs3*Cnnd[NsTs100*i + Ns100*j + 4*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 4*Ns + kd]
             +cs4*Cnnd[NsTs100*i + Ns100*j + 5*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 5*Ns + kd]
             +cs4*Cnnd[NsTs100*i + Ns100*j + 6*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 6*Ns + kd]
             +cs5*Cnnd[NsTs100*i + Ns100*j + 7*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 7*Ns + kd]
             +cs5*Cnnd[NsTs100*i + Ns100*j + 8*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 8*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if (lMax > 2) {
    double prel3 = PI*sqrt(8.0/(2.0*3.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 3*NsNs + shiftN] = prel3*(
              cs6*Cnnd[NsTs100*i + Ns100*j + 9*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 9*Ns + kd]
             +cs7*Cnnd[NsTs100*i + Ns100*j + 10*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 10*Ns + kd]
             +cs7*Cnnd[NsTs100*i + Ns100*j + 11*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 11*Ns + kd]
             +cs8*Cnnd[NsTs100*i + Ns100*j + 12*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 12*Ns + kd]
             +cs8*Cnnd[NsTs100*i + Ns100*j + 13*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 13*Ns + kd]
             +cs9*Cnnd[NsTs100*i + Ns100*j + 14*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 14*Ns + kd]
             +cs9*Cnnd[NsTs100*i + Ns100*j + 15*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 15*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if (lMax > 3) {
    double prel4 = PI*sqrt(8.0/(2.0*4.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 4*NsNs + shiftN] = prel4*(
              cs10*Cnnd[NsTs100*i + Ns100*j + 16*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 16*Ns + kd]
             +cs11*Cnnd[NsTs100*i + Ns100*j + 17*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 17*Ns + kd]
             +cs11*Cnnd[NsTs100*i + Ns100*j + 18*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 18*Ns + kd]
             +cs12*Cnnd[NsTs100*i + Ns100*j + 19*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 19*Ns + kd]
             +cs12*Cnnd[NsTs100*i + Ns100*j + 20*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 20*Ns + kd]
             +cs13*Cnnd[NsTs100*i + Ns100*j + 21*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 21*Ns + kd]
             +cs13*Cnnd[NsTs100*i + Ns100*j + 22*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 22*Ns + kd]
             +cs14*Cnnd[NsTs100*i + Ns100*j + 23*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 23*Ns + kd]
             +cs14*Cnnd[NsTs100*i + Ns100*j + 24*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 24*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 4) {
    double prel5 = PI*sqrt(8.0/(2.0*5.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 5*NsNs + shiftN] = prel5*(
              cs15*Cnnd[NsTs100*i + Ns100*j + 25*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 25*Ns + kd]
             +cs16*Cnnd[NsTs100*i + Ns100*j + 26*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 26*Ns + kd]
             +cs16*Cnnd[NsTs100*i + Ns100*j + 27*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 27*Ns + kd]
             +cs17*Cnnd[NsTs100*i + Ns100*j + 28*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 28*Ns + kd]
             +cs17*Cnnd[NsTs100*i + Ns100*j + 29*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 29*Ns + kd]
             +cs18*Cnnd[NsTs100*i + Ns100*j + 30*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 30*Ns + kd]
             +cs18*Cnnd[NsTs100*i + Ns100*j + 31*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 31*Ns + kd]
             +cs19*Cnnd[NsTs100*i + Ns100*j + 32*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 32*Ns + kd]
             +cs19*Cnnd[NsTs100*i + Ns100*j + 33*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 33*Ns + kd]
             +cs20*Cnnd[NsTs100*i + Ns100*j + 34*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 34*Ns + kd]
             +cs20*Cnnd[NsTs100*i + Ns100*j + 35*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 35*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if (lMax > 5) {
    double prel6 = PI*sqrt(8.0/(2.0*6.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 6*NsNs + shiftN] = prel6*(
              cs21*Cnnd[NsTs100*i + Ns100*j + 36*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 36*Ns + kd]
             +cs22*Cnnd[NsTs100*i + Ns100*j + 37*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 37*Ns + kd]
             +cs22*Cnnd[NsTs100*i + Ns100*j + 38*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 38*Ns + kd]
             +cs23*Cnnd[NsTs100*i + Ns100*j + 39*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 39*Ns + kd]
             +cs23*Cnnd[NsTs100*i + Ns100*j + 40*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 40*Ns + kd]
             +cs24*Cnnd[NsTs100*i + Ns100*j + 41*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 41*Ns + kd]
             +cs24*Cnnd[NsTs100*i + Ns100*j + 42*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 42*Ns + kd]
             +cs25*Cnnd[NsTs100*i + Ns100*j + 43*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 43*Ns + kd]
             +cs25*Cnnd[NsTs100*i + Ns100*j + 44*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 44*Ns + kd]
             +cs26*Cnnd[NsTs100*i + Ns100*j + 45*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 45*Ns + kd]
             +cs26*Cnnd[NsTs100*i + Ns100*j + 46*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 46*Ns + kd]
             +cs27*Cnnd[NsTs100*i + Ns100*j + 47*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 47*Ns + kd]
             +cs27*Cnnd[NsTs100*i + Ns100*j + 48*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 48*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if (lMax > 6) {
    double prel7 = PI*sqrt(8.0/(2.0*7.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 7*NsNs + shiftN] = prel7*(
              cs28*Cnnd[NsTs100*i + Ns100*j + 49*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 49*Ns + kd]
             +cs29*Cnnd[NsTs100*i + Ns100*j + 50*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 50*Ns + kd]
             +cs29*Cnnd[NsTs100*i + Ns100*j + 51*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 51*Ns + kd]
             +cs30*Cnnd[NsTs100*i + Ns100*j + 52*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 52*Ns + kd]
             +cs30*Cnnd[NsTs100*i + Ns100*j + 53*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 53*Ns + kd]
             +cs31*Cnnd[NsTs100*i + Ns100*j + 54*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 54*Ns + kd]
             +cs31*Cnnd[NsTs100*i + Ns100*j + 55*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 55*Ns + kd]
             +cs32*Cnnd[NsTs100*i + Ns100*j + 56*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 56*Ns + kd]
             +cs32*Cnnd[NsTs100*i + Ns100*j + 57*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 57*Ns + kd]
             +cs33*Cnnd[NsTs100*i + Ns100*j + 58*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 58*Ns + kd]
             +cs33*Cnnd[NsTs100*i + Ns100*j + 59*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 59*Ns + kd]
             +cs34*Cnnd[NsTs100*i + Ns100*j + 60*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 60*Ns + kd]
             +cs34*Cnnd[NsTs100*i + Ns100*j + 61*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 61*Ns + kd]
             +cs35*Cnnd[NsTs100*i + Ns100*j + 62*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 62*Ns + kd]
             +cs35*Cnnd[NsTs100*i + Ns100*j + 63*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 63*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if (lMax > 7) {
    double prel8 = PI*sqrt(8.0/(2.0*8.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 8*NsNs + shiftN] = prel8*(
              cs36*Cnnd[NsTs100*i + Ns100*j + 64*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 64*Ns + kd]
             +cs37*Cnnd[NsTs100*i + Ns100*j + 65*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 65*Ns + kd]
             +cs37*Cnnd[NsTs100*i + Ns100*j + 66*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 66*Ns + kd]
             +cs38*Cnnd[NsTs100*i + Ns100*j + 67*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 67*Ns + kd]
             +cs38*Cnnd[NsTs100*i + Ns100*j + 68*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 68*Ns + kd]
             +cs39*Cnnd[NsTs100*i + Ns100*j + 69*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 69*Ns + kd]
             +cs39*Cnnd[NsTs100*i + Ns100*j + 70*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 70*Ns + kd]
             +cs40*Cnnd[NsTs100*i + Ns100*j + 71*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 71*Ns + kd]
             +cs40*Cnnd[NsTs100*i + Ns100*j + 72*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 72*Ns + kd]
             +cs41*Cnnd[NsTs100*i + Ns100*j + 73*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 73*Ns + kd]
             +cs41*Cnnd[NsTs100*i + Ns100*j + 74*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 74*Ns + kd]
             +cs42*Cnnd[NsTs100*i + Ns100*j + 75*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 75*Ns + kd]
             +cs42*Cnnd[NsTs100*i + Ns100*j + 76*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 76*Ns + kd]
             +cs43*Cnnd[NsTs100*i + Ns100*j + 77*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 77*Ns + kd]
             +cs43*Cnnd[NsTs100*i + Ns100*j + 78*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 78*Ns + kd]
             +cs44*Cnnd[NsTs100*i + Ns100*j + 79*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 79*Ns + kd]
             +cs44*Cnnd[NsTs100*i + Ns100*j + 80*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 80*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if (lMax > 8) {
    double prel9 = PI*sqrt(8.0/(2.0*9.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 9*NsNs + shiftN] = prel9*(
              cs45*Cnnd[NsTs100*i + Ns100*j + 81*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 81*Ns + kd]
             +cs46*Cnnd[NsTs100*i + Ns100*j + 82*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 82*Ns + kd]
             +cs46*Cnnd[NsTs100*i + Ns100*j + 83*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 83*Ns + kd]
             +cs47*Cnnd[NsTs100*i + Ns100*j + 84*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 84*Ns + kd]
             +cs47*Cnnd[NsTs100*i + Ns100*j + 85*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 85*Ns + kd]
             +cs48*Cnnd[NsTs100*i + Ns100*j + 86*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 86*Ns + kd]
             +cs48*Cnnd[NsTs100*i + Ns100*j + 87*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 87*Ns + kd]
             +cs49*Cnnd[NsTs100*i + Ns100*j + 88*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 88*Ns + kd]
             +cs49*Cnnd[NsTs100*i + Ns100*j + 89*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 89*Ns + kd]
             +cs50*Cnnd[NsTs100*i + Ns100*j + 90*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 90*Ns + kd]
             +cs50*Cnnd[NsTs100*i + Ns100*j + 91*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 91*Ns + kd]
             +cs51*Cnnd[NsTs100*i + Ns100*j + 92*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 92*Ns + kd]
             +cs51*Cnnd[NsTs100*i + Ns100*j + 93*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 93*Ns + kd]
             +cs52*Cnnd[NsTs100*i + Ns100*j + 94*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 94*Ns + kd]
             +cs52*Cnnd[NsTs100*i + Ns100*j + 95*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 95*Ns + kd]
             +cs53*Cnnd[NsTs100*i + Ns100*j + 96*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 96*Ns + kd]
             +cs53*Cnnd[NsTs100*i + Ns100*j + 97*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 97*Ns + kd]
             +cs54*Cnnd[NsTs100*i + Ns100*j + 98*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 98*Ns + kd]
             +cs54*Cnnd[NsTs100*i + Ns100*j + 99*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 99*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  }
}
//=======================================================================
/**
 * Used to calculate the partial power spectrum.
 */
void getPCrossOver(double* soapMat, double* Cnnd, int Ns, int Ts, int Hs, int lMax){
  int NsTs100 = Ns*Ts*100;
  int Ns100 = Ns*100;
  int NsNs = (Ns*(Ns+1))/2;
  int NsNsLmax = NsNs*(lMax+1) ;
  int NsNsLmaxTs = NsNsLmax*getCrosNum(Ts);
  int shiftN = 0;
  int shiftT = 0;

  //  double   cs0  = pow(PIHalf,2);
  //  double   cs1  = pow(2.7206990464,2);
  //  double cs2  = 2*pow(1.9238247452,2); double   cs3  = pow(1.7562036828,2); double cs4  = 2*pow(4.3018029072,2);
  //  double cs5  = 2*pow(2.1509014536,2); double   cs6  = pow(2.0779682205,2); double cs7  = 2*pow(1.7995732672,2);
  //  double cs8  = 2*pow(5.6907503408,2); double cs9  = 2*pow(2.3232390981,2); double   cs10 = pow(0.5890486225,2);
  //  double cs11 = 2*pow(2.6343055241,2); double cs12 = 2*pow(1.8627352998,2); double cs13 = 2*pow(6.9697172942,2);
  //  double cs14 = 2*pow(2.4641671809,2); double   cs15 = pow(0.6512177548,2); double cs16 = 2*pow(1.7834332706,2);
  //  double cs17 = 2*pow(9.4370418280,2); double cs18 = 2*pow(1.9263280966,2); double cs19 = 2*pow(8.1727179596,2);
  //  double cs20 = 2*pow(2.5844403427,2); double   cs21 = pow(0.3539741687,2); double cs22 = 2*pow(2.2940148014,2);
  //  double cs23 = 2*pow(1.8135779397,2); double cs24 = 2*pow(3.6271558793,2); double cs25 = 2*pow(1.9866750947,2);
  //  double cs26 = 2*pow(9.3183321738,2); double cs27 = 2*pow(2.6899707945,2); double   cs28 = pow(0.3802292509,2);
  //  double cs29 = 2*pow(0.3556718963,2); double cs30 = 2*pow(0.8712146618,2); double cs31 = 2*pow(0.6160417952,2);
  //  double cs32 = 2*pow(4.0863589798,2); double cs33 = 2*pow(2.0431794899,2); double cs34 = 2*pow(10.418212089,2);
  //  double cs35 = 2*pow(2.7843843014,2); double   cs36 = pow(0.0505981185,2); double cs37 = 2*pow(0.4293392727,2);
  //  double cs38 = 2*pow(1.7960550366,2); double cs39 = 2*pow(4.8637400313,2); double cs40 = 2*pow(1.8837184141,2);
  //  double cs41 = 2*pow(13.583686661,2); double cs42 = 2*pow(2.0960083567,2); double cs43 = 2*pow(11.480310577,2);
  //  double cs44 = 2*pow(2.8700776442,2); double   cs45 = pow(0.0534917379,2); double cs46 = 2*pow(0.2537335916,2);
  //  double cs47 = 2*pow(2.3802320735,2); double cs48 = 2*pow(1.8179322747,2); double cs49 = 2*pow(16.055543121,2);
  //  double cs50 = 2*pow(1.9190044477,2); double cs51 = 2*pow(4.9548481782,2); double cs52 = 2*pow(2.1455121971,2);
  //  double cs53 = 2*pow(12.510378411,2); double cs54 = 2*pow(2.9487244699,2);
  double cs0=2.4674011003; double cs1=7.4022033011; double cs2=7.4022033005;
  double cs3=3.0842513755; double cs4=37.0110165048; double cs5=9.2527541262;
  double cs6=4.3179519254; double cs7=6.4769278880; double cs8=64.7692788826;
  double cs9=10.7948798139; double cs10=0.3469782797; double cs11=13.8791311886;
  double cs12=6.9395655942; double cs13=97.1539183221; double cs14=12.1442397908;
  double cs15=0.4240845642; double cs16=6.3612684614; double cs17=178.1155169268;
  double cs18=7.4214798715; double cs19=133.5866376943; double cs20=13.3586637700;
  double cs21=0.1252977121; double cs22=10.5250078181; double cs23=6.5781298867;
  double cs24=26.3125195455; double cs25=7.8937558638; double cs26=173.6626290026;
  double cs27=14.4718857505; double cs28=0.1445742832; double cs29=0.2530049956;
  double cs30=1.5180299739; double cs31=0.7590149869; double cs32=33.3966594236;
  double cs33=8.3491648559; double cs34=217.0782862628; double cs35=15.5055918758;
  double cs36=0.0025601696; double cs37=0.3686644222; double cs38=6.4516273890;
  double cs39=47.3119341841; double cs40=7.0967901272; double cs41=369.0330866085;
  double cs42=8.7865020627; double cs43=263.5950618888; double cs44=16.4746913675;
  double cs45=0.0028613660; double cs46=0.1287614710; double cs47=11.3310094474;
  double cs48=6.6097555108; double cs49=515.5609298206; double cs50=7.3651561406;
  double cs51=49.1010409380; double cs52=9.2064451758; double cs53=313.0191359728;
  double cs54=17.3899519988;

  // The power spectrum is multiplied by an l-dependent prefactor that comes
  // from the normalization of the Wigner D matrices. This prefactor is
  // mentioned in the arrata of the original SOAP paper: On representing
  // chemical environments, Phys. Rev. B 87, 184115 (2013). Here the square
  // root of the prefactor in the dot-product kernel is used, so that after a
  // possible dot-product the full prefactor is recovered.

  // SUM M's UP!
  double prel0 = PI*sqrt(8.0/(1.0));
  for(int i = 0; i < Hs; i++){
    shiftT = 0;
    for(int j = 0; j < Ts; j++){
      for(int jd = j; jd < Ts; jd++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i + NsNsLmax*shiftT + 0 + shiftN] = prel0*(
                cs0*Cnnd[NsTs100*i + Ns100*j + 0 + k]*Cnnd[NsTs100*i + Ns100*jd + 0 + kd]);
            shiftN++;
          }
        }
        shiftT++;
      }
    }
  } if(lMax > 0){
    double prel1 = PI*sqrt(8.0/(2.0*1.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ NsNs + shiftN] = prel1*(
                  cs1*Cnnd[NsTs100*i + Ns100*j + 1*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 1*Ns + kd]
                  +cs2*Cnnd[NsTs100*i + Ns100*j + 2*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 2*Ns + kd]
                  +cs2*Cnnd[NsTs100*i + Ns100*j + 3*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 3*Ns + kd]);
              shiftN++;
            }
          }
          shiftT++;
        }
      }
    }
  }  if(lMax > 1){
    double prel2 = PI*sqrt(8.0/(2.0*2.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 2*NsNs + shiftN] = prel2*(
                  cs3*Cnnd[NsTs100*i + Ns100*j + 4*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 4*Ns + kd]
                  +cs4*Cnnd[NsTs100*i + Ns100*j + 5*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 5*Ns + kd]
                  +cs4*Cnnd[NsTs100*i + Ns100*j + 6*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 6*Ns + kd]
                  +cs5*Cnnd[NsTs100*i + Ns100*j + 7*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 7*Ns + kd]
                  +cs5*Cnnd[NsTs100*i + Ns100*j + 8*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 8*Ns + kd]);
              shiftN++;
            }
          }
          shiftT++;
        }
      }
    }
  }  if(lMax > 2){
    double prel3 = PI*sqrt(8.0/(2.0*3.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 3*NsNs + shiftN] = prel3*(
                  cs6*Cnnd[NsTs100*i + Ns100*j + 9*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 9*Ns + kd]
                  +cs7*Cnnd[NsTs100*i + Ns100*j + 10*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 10*Ns + kd]
                  +cs7*Cnnd[NsTs100*i + Ns100*j + 11*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 11*Ns + kd]
                  +cs8*Cnnd[NsTs100*i + Ns100*j + 12*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 12*Ns + kd]
                  +cs8*Cnnd[NsTs100*i + Ns100*j + 13*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 13*Ns + kd]
                  +cs9*Cnnd[NsTs100*i + Ns100*j + 14*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 14*Ns + kd]
                  +cs9*Cnnd[NsTs100*i + Ns100*j + 15*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 15*Ns + kd]);
              shiftN++;
            }
          }
          shiftT++;
        }
      }
    }
  }  if(lMax > 3){
    double prel4 = PI*sqrt(8.0/(2.0*4.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 4*NsNs + shiftN] = prel4*(
                  cs10*Cnnd[NsTs100*i + Ns100*j + 16*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 16*Ns + kd]
                  +cs11*Cnnd[NsTs100*i + Ns100*j + 17*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 17*Ns + kd]
                  +cs11*Cnnd[NsTs100*i + Ns100*j + 18*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 18*Ns + kd]
                  +cs12*Cnnd[NsTs100*i + Ns100*j + 19*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 19*Ns + kd]
                  +cs12*Cnnd[NsTs100*i + Ns100*j + 20*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 20*Ns + kd]
                  +cs13*Cnnd[NsTs100*i + Ns100*j + 21*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 21*Ns + kd]
                  +cs13*Cnnd[NsTs100*i + Ns100*j + 22*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 22*Ns + kd]
                  +cs14*Cnnd[NsTs100*i + Ns100*j + 23*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 23*Ns + kd]
                  +cs14*Cnnd[NsTs100*i + Ns100*j + 24*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 24*Ns + kd]);
              shiftN++;
            }
          }
          shiftT++;
        }
      }
    }
  }  if(lMax > 4) {
    double prel5 = PI*sqrt(8.0/(2.0*5.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 5*NsNs + shiftN] = prel5*(
                  cs15*Cnnd[NsTs100*i + Ns100*j + 25*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 25*Ns + kd]
                  +cs16*Cnnd[NsTs100*i + Ns100*j + 26*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 26*Ns + kd]
                  +cs16*Cnnd[NsTs100*i + Ns100*j + 27*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 27*Ns + kd]
                  +cs17*Cnnd[NsTs100*i + Ns100*j + 28*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 28*Ns + kd]
                  +cs17*Cnnd[NsTs100*i + Ns100*j + 29*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 29*Ns + kd]
                  +cs18*Cnnd[NsTs100*i + Ns100*j + 30*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 30*Ns + kd]
                  +cs18*Cnnd[NsTs100*i + Ns100*j + 31*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 31*Ns + kd]
                  +cs19*Cnnd[NsTs100*i + Ns100*j + 32*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 32*Ns + kd]
                  +cs19*Cnnd[NsTs100*i + Ns100*j + 33*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 33*Ns + kd]
                  +cs20*Cnnd[NsTs100*i + Ns100*j + 34*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 34*Ns + kd]
                  +cs20*Cnnd[NsTs100*i + Ns100*j + 35*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 35*Ns + kd]);
              shiftN++;
            }
          }
          shiftT++;
        }
      }
    }
  }  if(lMax > 5){
    double prel6 = PI*sqrt(8.0/(2.0*6.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 6*NsNs + shiftN] = prel6*(
                  cs21*Cnnd[NsTs100*i + Ns100*j + 36*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 36*Ns + kd]
                  +cs22*Cnnd[NsTs100*i + Ns100*j + 37*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 37*Ns + kd]
                  +cs22*Cnnd[NsTs100*i + Ns100*j + 38*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 38*Ns + kd]
                  +cs23*Cnnd[NsTs100*i + Ns100*j + 39*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 39*Ns + kd]
                  +cs23*Cnnd[NsTs100*i + Ns100*j + 40*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 40*Ns + kd]
                  +cs24*Cnnd[NsTs100*i + Ns100*j + 41*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 41*Ns + kd]
                  +cs24*Cnnd[NsTs100*i + Ns100*j + 42*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 42*Ns + kd]
                  +cs25*Cnnd[NsTs100*i + Ns100*j + 43*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 43*Ns + kd]
                  +cs25*Cnnd[NsTs100*i + Ns100*j + 44*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 44*Ns + kd]
                  +cs26*Cnnd[NsTs100*i + Ns100*j + 45*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 45*Ns + kd]
                  +cs26*Cnnd[NsTs100*i + Ns100*j + 46*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 46*Ns + kd]
                  +cs27*Cnnd[NsTs100*i + Ns100*j + 47*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 47*Ns + kd]
                  +cs27*Cnnd[NsTs100*i + Ns100*j + 48*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 48*Ns + kd]);
              shiftN++;
            }
          }
          shiftT++;
        }
      }
    }
  }  if(lMax > 6){
    double prel7 = PI*sqrt(8.0/(2.0*7.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 7*NsNs + shiftN] = prel7*(
                  cs28*Cnnd[NsTs100*i + Ns100*j + 49*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 49*Ns + kd]
                  +cs29*Cnnd[NsTs100*i + Ns100*j + 50*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 50*Ns + kd]
                  +cs29*Cnnd[NsTs100*i + Ns100*j + 51*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 51*Ns + kd]
                  +cs30*Cnnd[NsTs100*i + Ns100*j + 52*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 52*Ns + kd]
                  +cs30*Cnnd[NsTs100*i + Ns100*j + 53*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 53*Ns + kd]
                  +cs31*Cnnd[NsTs100*i + Ns100*j + 54*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 54*Ns + kd]
                  +cs31*Cnnd[NsTs100*i + Ns100*j + 55*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 55*Ns + kd]
                  +cs32*Cnnd[NsTs100*i + Ns100*j + 56*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 56*Ns + kd]
                  +cs32*Cnnd[NsTs100*i + Ns100*j + 57*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 57*Ns + kd]
                  +cs33*Cnnd[NsTs100*i + Ns100*j + 58*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 58*Ns + kd]
                  +cs33*Cnnd[NsTs100*i + Ns100*j + 59*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 59*Ns + kd]
                  +cs34*Cnnd[NsTs100*i + Ns100*j + 60*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 60*Ns + kd]
                  +cs34*Cnnd[NsTs100*i + Ns100*j + 61*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 61*Ns + kd]
                  +cs35*Cnnd[NsTs100*i + Ns100*j + 62*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 62*Ns + kd]
                  +cs35*Cnnd[NsTs100*i + Ns100*j + 63*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 63*Ns + kd]);
              shiftN++;
            }
          }
          shiftT++;
        }
      }
    }
  }  if(lMax > 7){
    double prel8 = PI*sqrt(8.0/(2.0*8.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 8*NsNs + shiftN] = prel8*(
                  cs36*Cnnd[NsTs100*i + Ns100*j + 64*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 64*Ns + kd]
                  +cs37*Cnnd[NsTs100*i + Ns100*j + 65*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 65*Ns + kd]
                  +cs37*Cnnd[NsTs100*i + Ns100*j + 66*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 66*Ns + kd]
                  +cs38*Cnnd[NsTs100*i + Ns100*j + 67*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 67*Ns + kd]
                  +cs38*Cnnd[NsTs100*i + Ns100*j + 68*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 68*Ns + kd]
                  +cs39*Cnnd[NsTs100*i + Ns100*j + 69*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 69*Ns + kd]
                  +cs39*Cnnd[NsTs100*i + Ns100*j + 70*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 70*Ns + kd]
                  +cs40*Cnnd[NsTs100*i + Ns100*j + 71*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 71*Ns + kd]
                  +cs40*Cnnd[NsTs100*i + Ns100*j + 72*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 72*Ns + kd]
                  +cs41*Cnnd[NsTs100*i + Ns100*j + 73*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 73*Ns + kd]
                  +cs41*Cnnd[NsTs100*i + Ns100*j + 74*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 74*Ns + kd]
                  +cs42*Cnnd[NsTs100*i + Ns100*j + 75*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 75*Ns + kd]
                  +cs42*Cnnd[NsTs100*i + Ns100*j + 76*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 76*Ns + kd]
                  +cs43*Cnnd[NsTs100*i + Ns100*j + 77*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 77*Ns + kd]
                  +cs43*Cnnd[NsTs100*i + Ns100*j + 78*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 78*Ns + kd]
                  +cs44*Cnnd[NsTs100*i + Ns100*j + 79*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 79*Ns + kd]
                  +cs44*Cnnd[NsTs100*i + Ns100*j + 80*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 80*Ns + kd]);
              shiftN++;
            }
          }
          shiftT++;
        }
      }
    }
  }  if(lMax > 8){
    double prel9 = PI*sqrt(8.0/(2.0*9.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 9*NsNs + shiftN] = prel9*(
                  cs45*Cnnd[NsTs100*i + Ns100*j + 81*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 81*Ns + kd]
                  +cs46*Cnnd[NsTs100*i + Ns100*j + 82*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 82*Ns + kd]
                  +cs46*Cnnd[NsTs100*i + Ns100*j + 83*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 83*Ns + kd]
                  +cs47*Cnnd[NsTs100*i + Ns100*j + 84*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 84*Ns + kd]
                  +cs47*Cnnd[NsTs100*i + Ns100*j + 85*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 85*Ns + kd]
                  +cs48*Cnnd[NsTs100*i + Ns100*j + 86*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 86*Ns + kd]
                  +cs48*Cnnd[NsTs100*i + Ns100*j + 87*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 87*Ns + kd]
                  +cs49*Cnnd[NsTs100*i + Ns100*j + 88*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 88*Ns + kd]
                  +cs49*Cnnd[NsTs100*i + Ns100*j + 89*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 89*Ns + kd]
                  +cs50*Cnnd[NsTs100*i + Ns100*j + 90*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 90*Ns + kd]
                  +cs50*Cnnd[NsTs100*i + Ns100*j + 91*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 91*Ns + kd]
                  +cs51*Cnnd[NsTs100*i + Ns100*j + 92*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 92*Ns + kd]
                  +cs51*Cnnd[NsTs100*i + Ns100*j + 93*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 93*Ns + kd]
                  +cs52*Cnnd[NsTs100*i + Ns100*j + 94*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 94*Ns + kd]
                  +cs52*Cnnd[NsTs100*i + Ns100*j + 95*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 95*Ns + kd]
                  +cs53*Cnnd[NsTs100*i + Ns100*j + 96*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 96*Ns + kd]
                  +cs53*Cnnd[NsTs100*i + Ns100*j + 97*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 97*Ns + kd]
                  +cs54*Cnnd[NsTs100*i + Ns100*j + 98*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 98*Ns + kd]
                  +cs54*Cnnd[NsTs100*i + Ns100*j + 99*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 99*Ns + kd]);
              shiftN++;
            }
          }
          shiftT++;
        }
      }
    }
  }
}

//===========================================================================================
void soapGTO(py::array_t<double> cArr, py::array_t<double> positions, py::array_t<double> HposArr, py::array_t<double> alphasArr, py::array_t<double> betasArr, py::array_t<int> atomicNumbersArr, double rCut, double cutoffPadding, int totalAN, int Nt, int Ns, int lMax, int Hs, double eta, bool crossover) {

  auto atomicNumbers = atomicNumbersArr.unchecked<1>();
  double *c = (double*)cArr.request().ptr;
  double *Hpos = (double*)HposArr.request().ptr;
  double *alphas = (double*)alphasArr.request().ptr;
  double *betas = (double*)betasArr.request().ptr;

  double oOeta = 1.0/eta;
  double oOeta3O2 = sqrt(oOeta*oOeta*oOeta);

  double NsNs = Ns*Ns;
  double* dx  = (double*) malloc(sizeof(double)*totalAN);
  double* dy  = (double*) malloc(sizeof(double)*totalAN);
  double* dz  = (double*) malloc(sizeof(double)*totalAN);
  double* z2 = (double*) malloc(sizeof(double)*totalAN);
  double* z4 = (double*) malloc(sizeof(double)*totalAN);
  double* z6 = (double*) malloc(sizeof(double)*totalAN);
  double* z8 = (double*) malloc(sizeof(double)*totalAN);
  double* r2 = (double*) malloc(sizeof(double)*totalAN);
  double* r4 = (double*) malloc(sizeof(double)*totalAN);
  double* r6 = (double*) malloc(sizeof(double)*totalAN);
  double* r8 = (double*) malloc(sizeof(double)*totalAN);
  double* ReIm2 = (double*) malloc(2*sizeof(double)*totalAN);// 2 -> Re + ixIm
  double* ReIm3 = (double*) malloc(2*sizeof(double)*totalAN);// 2 -> Re + ixIm
  double* ReIm4 = (double*) malloc(2*sizeof(double)*totalAN);// 2 -> Re + ixIm
  double* ReIm5 = (double*) malloc(2*sizeof(double)*totalAN);// 2 -> Re + ixIm
  double* ReIm6 = (double*) malloc(2*sizeof(double)*totalAN);// 2 -> Re + ixIm
  double* ReIm7 = (double*) malloc(2*sizeof(double)*totalAN);// 2 -> Re + ixIm
  double* ReIm8 = (double*) malloc(2*sizeof(double)*totalAN);// 2 -> Re + ixIm
  double* ReIm9 = (double*) malloc(2*sizeof(double)*totalAN);// 2 -> Re + ixIm
  double* exes = (double*) malloc (sizeof(double)*totalAN);
  double* preCoef = (double*) malloc(96*sizeof(double)*totalAN);
  double* bOa = (double*) malloc((lMax+1)*NsNs*sizeof(double));
  double* aOa = (double*) malloc((lMax+1)*Ns*sizeof(double));

  int Nx2 = 2*Ns; int Nx3 = 3*Ns; int Nx4 = 4*Ns; int Nx5 = 5*Ns;
  int Nx6 = 6*Ns; int Nx7 = 7*Ns; int Nx8 = 8*Ns; int Nx9 = 9*Ns;
  int Nx10 = 10*Ns; int Nx11 = 11*Ns; int Nx12 = 12*Ns; int Nx13 = 13*Ns;
  int Nx14 = 14*Ns; int Nx15 = 15*Ns; int Nx16 = 16*Ns; int Nx17 = 17*Ns;
  int Nx18 = 18*Ns; int Nx19 = 19*Ns; int Nx20 = 20*Ns; int Nx21 = 21*Ns;
  int Nx22 = 22*Ns; int Nx23 = 23*Ns; int Nx24 = 24*Ns; int Nx25 = 25*Ns;
  int Nx26 = 26*Ns; int Nx27 = 27*Ns; int Nx28 = 28*Ns; int Nx29 = 29*Ns;
  int Nx30 = 30*Ns; int Nx31 = 31*Ns; int Nx32 = 32*Ns; int Nx33 = 33*Ns;
  int Nx34 = 34*Ns; int Nx35 = 35*Ns; int Nx36 = 36*Ns; int Nx37 = 37*Ns;
  int Nx38 = 38*Ns; int Nx39 = 39*Ns; int Nx40 = 40*Ns; int Nx41 = 41*Ns;
  int Nx42 = 42*Ns; int Nx43 = 43*Ns; int Nx44 = 44*Ns; int Nx45 = 45*Ns;
  int Nx46 = 46*Ns; int Nx47 = 47*Ns; int Nx48 = 48*Ns; int Nx49 = 49*Ns;
  int Nx50 = 50*Ns; int Nx51 = 51*Ns; int Nx52 = 52*Ns; int Nx53 = 53*Ns;
  int Nx54 = 54*Ns; int Nx55 = 55*Ns; int Nx56 = 56*Ns; int Nx57 = 57*Ns;
  int Nx58 = 58*Ns; int Nx59 = 59*Ns; int Nx60 = 60*Ns; int Nx61 = 61*Ns;
  int Nx62 = 62*Ns; int Nx63 = 63*Ns; int Nx64 = 64*Ns; int Nx65 = 65*Ns;
  int Nx66 = 66*Ns; int Nx67 = 67*Ns; int Nx68 = 68*Ns; int Nx69 = 69*Ns;
  int Nx70 = 70*Ns; int Nx71 = 71*Ns; int Nx72 = 72*Ns; int Nx73 = 73*Ns;
  int Nx74 = 74*Ns; int Nx75 = 75*Ns; int Nx76 = 76*Ns; int Nx77 = 77*Ns;
  int Nx78 = 78*Ns; int Nx79 = 79*Ns; int Nx80 = 80*Ns; int Nx81 = 81*Ns;
  int Nx82 = 82*Ns; int Nx83 = 83*Ns; int Nx84 = 84*Ns; int Nx85 = 85*Ns;
  int Nx86 = 86*Ns; int Nx87 = 87*Ns; int Nx88 = 88*Ns; int Nx89 = 89*Ns;
  int Nx90 = 90*Ns; int Nx91 = 91*Ns; int Nx92 = 92*Ns; int Nx93 = 93*Ns;
  int Nx94 = 94*Ns; int Nx95 = 95*Ns; int Nx96 = 96*Ns; int Nx97 = 97*Ns;
  int Nx98 = 98*Ns; int Nx99 = 99*Ns;
  int t2 = 2*totalAN;  int t3 = 3*totalAN;  int t4 = 4*totalAN;
  int t5 = 5*totalAN;  int t6 = 6*totalAN;  int t7 = 7*totalAN;
  int t8 = 8*totalAN;  int t9 = 9*totalAN;  int t10 = 10*totalAN;
  int t11 = 11*totalAN;  int t12 = 12*totalAN;  int t13 = 13*totalAN;
  int t14 = 14*totalAN;  int t15 = 15*totalAN;  int t16 = 16*totalAN;
  int t17 = 17*totalAN;  int t18 = 18*totalAN;  int t19 = 19*totalAN;
  int t20 = 20*totalAN;  int t21 = 21*totalAN;  int t22 = 22*totalAN;
  int t23 = 23*totalAN;  int t24 = 24*totalAN;  int t25 = 25*totalAN;
  int t26 = 26*totalAN;  int t27 = 27*totalAN;  int t28 = 28*totalAN;
  int t29 = 29*totalAN;  int t30 = 30*totalAN;  int t31 = 31*totalAN;
  int t32 = 32*totalAN;  int t33 = 33*totalAN;  int t34 = 34*totalAN;
  int t35 = 35*totalAN;  int t36 = 36*totalAN;  int t37 = 37*totalAN;
  int t38 = 38*totalAN;  int t39 = 39*totalAN;  int t40 = 40*totalAN;
  int t41 = 41*totalAN;  int t42 = 42*totalAN;  int t43 = 43*totalAN;
  int t44 = 44*totalAN;  int t45 = 45*totalAN;  int t46 = 46*totalAN;
  int t47 = 47*totalAN;  int t48 = 48*totalAN;  int t49 = 49*totalAN;
  int t50 = 50*totalAN;  int t51 = 51*totalAN;  int t52 = 52*totalAN;
  int t53 = 53*totalAN;  int t54 = 54*totalAN;  int t55 = 55*totalAN;
  int t56 = 56*totalAN;  int t57 = 57*totalAN;  int t58 = 58*totalAN;
  int t59 = 59*totalAN;  int t60 = 60*totalAN;  int t61 = 61*totalAN;
  int t62 = 62*totalAN;  int t63 = 63*totalAN;  int t64 = 64*totalAN;
  int t65 = 65*totalAN;  int t66 = 66*totalAN;  int t67 = 67*totalAN;
  int t68 = 68*totalAN;  int t69 = 69*totalAN;  int t70 = 70*totalAN;
  int t71 = 71*totalAN;  int t72 = 72*totalAN;  int t73 = 73*totalAN;
  int t74 = 74*totalAN;  int t75 = 75*totalAN;  int t76 = 76*totalAN;
  int t77 = 77*totalAN;  int t78 = 78*totalAN;  int t79 = 79*totalAN;
  int t80 = 80*totalAN;  int t81 = 81*totalAN;  int t82 = 82*totalAN;
  int t83 = 83*totalAN;  int t84 = 84*totalAN;  int t85 = 85*totalAN;
  int t86 = 86*totalAN;  int t87 = 87*totalAN;  int t88 = 88*totalAN;
  int t89 = 89*totalAN;  int t90 = 90*totalAN;  int t91 = 91*totalAN;
  int t92 = 92*totalAN;  int t93 = 93*totalAN;  int t94 = 94*totalAN;
  int t95 = 95*totalAN;  int t96 = 96*totalAN;  int t97 = 97*totalAN;
  int t98 = 98*totalAN;  int t99 = 99*totalAN;

  double* cnnd = (double*) malloc(100*Nt*Ns*Hs*sizeof(double));
  for(int i = 0; i < 100*Nt*Ns*Hs; i++){cnnd[i] = 0.0;}

  // Initialize binning
  CellList cellList(positions, rCut+cutoffPadding);

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

  getAlphaBeta(aOa,bOa,alphas,betas,Ns,lMax,oOeta, oOeta3O2);

  // Loop through the centers
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

      // Save the neighbour distances into the arrays dx, dy and dz
      getDeltas(dx, dy, dz, positions, ix, iy, iz, ZIndexPair.second);

      getRsZs(dx, dy, dz, r2, r4, r6, r8, z2, z4, z6, z8, n_neighbours);
      getCfactors(preCoef, n_neighbours, dx, dy, dz, z2, z4, z6, z8, r2, r4, r6, r8, ReIm2, ReIm3, ReIm4, ReIm5, ReIm6, ReIm7, ReIm8, ReIm9, totalAN, lMax, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40, t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60, t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75, t76, t77, t78, t79, t80, t81, t82, t83, t84, t85, t86, t87, t88, t89, t90, t91, t92, t93, t94, t95, t96, t97, t98, t99);
      getC(cnnd, preCoef, dx, dy, dz, r2, bOa, aOa, exes, totalAN, n_neighbours, Ns, Nt, lMax, i, j, Nx2, Nx3, Nx4, Nx5, Nx6, Nx7, Nx8, Nx9, Nx10, Nx11, Nx12, Nx13, Nx14, Nx15, Nx16, Nx17, Nx18, Nx19, Nx20, Nx21, Nx22, Nx23, Nx24, Nx25, Nx26, Nx27, Nx28, Nx29, Nx30, Nx31, Nx32, Nx33, Nx34, Nx35, Nx36, Nx37, Nx38, Nx39, Nx40, Nx41, Nx42, Nx43, Nx44, Nx45, Nx46, Nx47, Nx48, Nx49, Nx50, Nx51, Nx52, Nx53, Nx54, Nx55, Nx56, Nx57, Nx58, Nx59, Nx60, Nx61, Nx62, Nx63, Nx64, Nx65, Nx66, Nx67, Nx68, Nx69, Nx70, Nx71, Nx72, Nx73, Nx74, Nx75, Nx76, Nx77, Nx78, Nx79, Nx80, Nx81, Nx82, Nx83, Nx84, Nx85, Nx86, Nx87, Nx88, Nx89, Nx90, Nx91, Nx92, Nx93, Nx94, Nx95, Nx96, Nx97, Nx98, Nx99, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40, t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, t52, t53, t54, t55, t56, t57, t58, t59, t60, t61, t62, t63, t64, t65, t66, t67, t68, t69, t70, t71, t72, t73, t74, t75, t76, t77, t78, t79, t80, t81, t82, t83, t84, t85, t86, t87, t88, t89, t90, t91, t92, t93, t94, t95, t96, t97, t98, t99);
    }
  }

  free(dx);
  free(dy);
  free(dz);
  free(z2);
  free(z4);
  free(z6);
  free(z8);
  free(r2);
  free(r4);
  free(r6);
  free(r8);
  free(ReIm2);
  free(ReIm3);
  free(ReIm4);
  free(ReIm5);
  free(ReIm6);
  free(ReIm7);
  free(ReIm8);
  free(ReIm9);
  free(exes);
  free(preCoef);
  free(bOa);
  free(aOa);

  if (crossover) {
    getPCrossOver(c, cnnd, Ns, Nt, Hs, lMax);
  } else {
    getPNoCross(c, cnnd, Ns, Nt, Hs, lMax);
  };
  free(cnnd);

  return;
}
