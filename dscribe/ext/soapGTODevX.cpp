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
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <map>
#include <set>
#include "soapGTODevX.h"
#include "celllist.h"

#define PI2 9.86960440108936
#define PI 3.14159265359
#define PI3 31.00627668029982 
#define PIHalf 1.57079632679490
//===========================================================
inline int getCrosNumD(int n)
{
  return n*(n+1)/2;
}
//================================================================
inline void getDeltaD(double* x, double* y, double* z, const py::array_t<double> &positions, const double ix, const double iy, const double iz, const vector<int> &indices){

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
inline void getRsZsD(double* x,double* x2,double* x4,double* x6,double* x8,double* x10,double* x12,double* x14,double* x16,double* x18, double* y,double* y2,double* y4,double* y6,double* y8,double* y10,double* y12,double* y14,double* y16,double* y18, double* z,double* r2,double* r4,double* r6,double* r8,double* r10,double* r12,double* r14,double* r16,double* r18,double* z2,double* z4,double* z6,double* z8,double* z10,double* z12,double* z14,double* z16,double* z18, int size, int lMax){
  for(int i = 0; i < size; i++){

    r2[i] = x[i]*x[i] + y[i]*y[i] + z[i]*z[i];
    z2[i] = z[i]*z[i]; x2[i] = x[i]*x[i]; y2[i] = y[i]*y[i];

    if(lMax > 3){
      r4[i] = r2[i]*r2[i];
      z4[i] = z2[i]*z2[i];
      x4[i] = x2[i]*x2[i];
      y4[i] = y2[i]*y2[i];
      if(lMax > 5){
        r6[i] = r2[i]*r4[i];
        z6[i] = z2[i]*z4[i];
        x6[i] = x2[i]*x4[i];
        y6[i] = y2[i]*y4[i];
        if(lMax > 7){
          r8[i] = r4[i]*r4[i];
          z8[i] = z4[i]*z4[i];
          x8[i] = x4[i]*x4[i];
          y8[i] = y4[i]*y4[i];
	}
      }
    }
    if(lMax > 9){ // suddenly x,y,z^n because switching to tesseral.
      x10[i] = x6[i]*x4[i];
      y10[i] = y6[i]*y4[i];
      z10[i] = z6[i]*z4[i];
      r10[i] = r6[i]*r4[i];
      if(lMax > 11){
        x12[i] = x6[i]*x6[i];
        y12[i] = y6[i]*y6[i];
        r12[i] = r6[i]*r6[i];
        z12[i] = z6[i]*z6[i];
        if(lMax > 13){
          x14[i] = x6[i]*x8[i];
          y14[i] = y6[i]*y8[i];
          r14[i] = r6[i]*r8[i];
          z14[i] = z6[i]*z8[i];
          if(lMax > 15){
            x16[i] = x8[i]*x8[i];
            y16[i] = y8[i]*y8[i];
            r16[i] = r8[i]*r8[i];
            z16[i] = z8[i]*z8[i];
            if(lMax > 17){
              x18[i] = x10[i]*x8[i];
              y18[i] = y10[i]*y8[i];
              r18[i] = r10[i]*r8[i];
              z18[i] = z10[i]*z8[i];
	    }
    	  }
        }
      }
    }
  }
}
//================================================================
void getAlphaBetaD(double* aOa, double* bOa, double* alphas, double* betas, int Ns,int lMax, double oOeta, double oOeta3O2){

  int  NsNs = Ns*Ns;
  double  oneO1alpha;  double  oneO1alpha2; double  oneO1alpha3;
  double  oneO1alpha4; double  oneO1alpha5; double  oneO1alpha6;
  double  oneO1alpha7; double  oneO1alpha8; double  oneO1alpha9;
  double  oneO1alpha10;
  double  oneO1alpha11;
  double  oneO1alpha12;
  double  oneO1alpha13;
  double  oneO1alpha14;
  double  oneO1alpha15;
  double  oneO1alpha16;
  double  oneO1alpha17;
  double  oneO1alpha18;
  double  oneO1alpha19;
  double  oneO1alpha20;
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
  }
  if(lMax > 8){
    int shift1 = 9*Ns; int shift2 = 9*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_9k
      oneO1alpha10 = pow(oneO1alpha,10); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha10;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;} // got beta_9nk
    }
  }
  if(lMax > 9){ //OBS!!!!! lMax > 9
    int shift1 = 10*Ns; int shift2 = 10*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_10k
      oneO1alpha11 = pow(oneO1alpha,11); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha11;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;
      } // got beta_10nk ... check, segfault
    }
  }

  if(lMax > 10){ //OBS!!!!! lMax > 9
    int shift1 = 11*Ns; int shift2 = 11*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_11k
      oneO1alpha12 = pow(oneO1alpha,12); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha12;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;
      } // got beta_10nk ... check, segfault
    }
  }
  if(lMax > 11){ //OBS!!!!! lMax > 9
    int shift1 = 12*Ns; int shift2 = 12*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_12k
      oneO1alpha13 = pow(oneO1alpha,13); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha13;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;
      } // got beta_10nk ... check, segfault
    }
  }
  if(lMax > 12){ //OBS!!!!! lMax > 9
    int shift1 = 13*Ns; int shift2 = 13*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_13k
      oneO1alpha14 = pow(oneO1alpha,14); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha14;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;
      } // got beta_10nk ... check, segfault
    }
  }
  if(lMax > 13){ //OBS!!!!! lMax > 9
    int shift1 = 14*Ns; int shift2 = 14*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_14k
      oneO1alpha15 = pow(oneO1alpha,15); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha15;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;
      } // got beta_10nk ... check, segfault
    }
  }
  if(lMax > 14){ //OBS!!!!! lMax > 9
    int shift1 = 15*Ns; int shift2 = 15*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_15k
      oneO1alpha16 = pow(oneO1alpha,16); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha16;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;
      } // got beta_10nk ... check, segfault
    }
  }
  if(lMax > 15){ //OBS!!!!! lMax > 9
    int shift1 = 16*Ns; int shift2 = 16*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_16k
      oneO1alpha17 = pow(oneO1alpha,17); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha17;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;
      } // got beta_10nk ... check, segfault
    }
  }
  if(lMax > 16){ //OBS!!!!! lMax > 9
    int shift1 = 17*Ns; int shift2 = 17*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_17k
      oneO1alpha18 = pow(oneO1alpha,18); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha18;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;
      } // got beta_10nk ... check, segfault
    }
  }
  if(lMax > 17){ //OBS!!!!! lMax > 9
    int shift1 = 18*Ns; int shift2 = 18*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_18k
      oneO1alpha19 = pow(oneO1alpha,19); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha19;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;
      } // got beta_10nk ... check, segfault
    }
  }
  if(lMax > 18){ //OBS!!!!! lMax > 9
    int shift1 = 19*Ns; int shift2 = 19*NsNs;
    for(int k = 0; k < Ns; k++){
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[shift1 + k]); oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[shift1 + k] = -alphas[shift1 + k]*oneO1alpha; //got alpha_19k
      oneO1alpha20 = pow(oneO1alpha,20); oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha20;
      for(int n = 0; n < Ns; n++){bOa[shift2 + n*Ns + k] = oOeta3O2*betas[shift2 + n*Ns + k]*oneO1alphaSqrtX;
      } // got beta_10nk ... check, segfault
    }
  }
}
//================================================================
void getCfactorsD(double* preCoef, int Asize, double* x,double* x2, double* x4, double* x6, double* x8, double* x10,double* x12,double* x14,double* x16,double* x18, double* y,double* y2, double* y4, double* y6, double* y8, double* y10,double* y12,double* y14,double* y16,double* y18, double* z, double* z2, double* z4, double* z6, double* z8, double* z10,double* z12,double* z14,double* z16,double* z18, double* r2, double* r4, double* r6, double* r8,double* r10,double* r12,double* r14,double* r16,double* r18,int totalAN, int lMax){

  for (int i = 0; i < Asize; i++) {
    if (lMax > 1){
      /*c20  */  preCoef[         +i] = 1.09254843059208*x[i]*y[i];
      /*c21Re*/  preCoef[totalAN*1+i] = 1.09254843059208*y[i]*z[i];
      /*c21Im*/  preCoef[totalAN*2+i] = -0.31539156525252*x2[i] - 0.31539156525252*y2[i] + 0.63078313050504*z2[i];
      /*c22Re*/  preCoef[totalAN*3+i] = 1.09254843059208*x[i]*z[i];
      /*c22Im*/  preCoef[totalAN*4+i] = 0.54627421529604*x2[i] - 0.54627421529604*y2[i];
    }
    if (lMax > 2){
      /*c30  */  preCoef[totalAN*5+i] = 0.590043589926644*y[i]*(3.0*x2[i] - y2[i]);
      /*c31Re*/  preCoef[totalAN*6+i] = 2.89061144264055*x[i]*y[i]*z[i];
      /*c31Im*/  preCoef[totalAN*7+i] = -0.457045799464466*y[i]*(x2[i] + y2[i] - 4.0*z2[i]);
      /*c32Re*/  preCoef[totalAN*8+i] = 0.373176332590115*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 2.0*z2[i]);
      /*c32Im*/  preCoef[totalAN*9+i] = -0.457045799464466*x[i]*(x2[i] + y2[i] - 4.0*z2[i]);
      /*c33Re*/  preCoef[totalAN*10+i] = 1.44530572132028*z[i]*(x2[i] - y2[i]);
      /*c33Im*/  preCoef[totalAN*11+i] = 0.590043589926644*x[i]*(x2[i] - 3.0*y2[i]);
    }
    if (lMax > 3){
      /*c40  */  preCoef[totalAN*12+i] = 2.5033429417967*x[i]*y[i]*(x2[i] - y2[i]);
      /*c41Re*/  preCoef[totalAN*13+i] = 1.77013076977993*y[i]*z[i]*(3.0*x2[i] - y2[i]);
      /*c41Im*/  preCoef[totalAN*14+i] = -0.94617469575756*x[i]*y[i]*(x2[i] + y2[i] - 6.0*z2[i]);
      /*c42Re*/  preCoef[totalAN*15+i] = 0.669046543557289*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 4.0*z2[i]);
      /*c42Im*/  preCoef[totalAN*16+i] = 3.70249414203215*z4[i] - 3.17356640745613*z2[i]*r2[i] + 0.317356640745613*r4[i];
      /*c43Re*/  preCoef[totalAN*17+i] = 0.669046543557289*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 4.0*z2[i]);
      /*c43Im*/  preCoef[totalAN*18+i] = -0.47308734787878*(x2[i] - y2[i])*(x2[i] + y2[i] - 6.0*z2[i]);
      /*c44Re*/  preCoef[totalAN*19+i] = 1.77013076977993*x[i]*z[i]*(x2[i] - 3.0*y2[i]);
      /*c44Im*/  preCoef[totalAN*20+i] = 0.625835735449176*x4[i] - 3.75501441269506*x2[i]*y2[i] + 0.625835735449176*y4[i];
    }
    if (lMax > 4){
      /*c50  */  preCoef[totalAN*21+i] = 0.65638205684017*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
      /*c51Re*/  preCoef[totalAN*22+i] = 8.30264925952416*x[i]*y[i]*z[i]*(x2[i] - y2[i]);
      /*c51Im*/  preCoef[totalAN*23+i] = -0.48923829943525*y[i]*(3.0*x2[i] - y2[i])*(x2[i] + y2[i] - 8.0*z2[i]);
      /*c52Re*/  preCoef[totalAN*24+i] = 4.79353678497332*x[i]*y[i]*z[i]*(-x2[i] - y2[i] + 2.0*z2[i]);
      /*c52Im*/  preCoef[totalAN*25+i] = 0.452946651195697*y[i]*(21.0*z4[i] - 14.0*z2[i]*r2[i] + r4[i]);
      /*c53Re*/  preCoef[totalAN*26+i] = 0.116950322453424*z[i]*(63.0*z4[i] - 70.0*z2[i]*r2[i] + 15.0*r4[i]);
      /*c53Im*/  preCoef[totalAN*27+i] = 0.452946651195697*x[i]*(21.0*z4[i] - 14.0*z2[i]*r2[i] + r4[i]);
      /*c54Re*/  preCoef[totalAN*28+i] = 2.39676839248666*z[i]*(x2[i] - y2[i])*(-x2[i] - y2[i] + 2.0*z2[i]);
      /*c54Im*/  preCoef[totalAN*29+i] = -0.48923829943525*x[i]*(x2[i] - 3.0*y2[i])*(x2[i] + y2[i] - 8.0*z2[i]);
      /*c55Re*/  preCoef[totalAN*30+i] = 2.07566231488104*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
      /*c55Im*/  preCoef[totalAN*31+i] = 0.65638205684017*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
    }
    if (lMax > 5){
      /*c60  */  preCoef[totalAN*32+i] = 1.36636821038383*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
      /*c61Re*/  preCoef[totalAN*33+i] = 2.36661916223175*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
      /*c61Im*/  preCoef[totalAN*34+i] = -2.0182596029149*x[i]*y[i]*(x2[i] - y2[i])*(x2[i] + y2[i] - 10.0*z2[i]);
      /*c62Re*/  preCoef[totalAN*35+i] = 0.921205259514923*y[i]*z[i]*(3.0*x2[i] - y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 8.0*z2[i]);
      /*c62Im*/  preCoef[totalAN*36+i] = 0.921205259514923*x[i]*y[i]*(33.0*z4[i] - 18.0*z2[i]*r2[i] + r4[i]);
      /*c63Re*/  preCoef[totalAN*37+i] = 0.582621362518731*y[i]*z[i]*(33.0*z4[i] - 30.0*z2[i]*r2[i] + 5.0*r4[i]);
      /*c63Im*/  preCoef[totalAN*38+i] = 14.6844857238222*z6[i] - 20.024298714303*z4[i]*r2[i] + 6.67476623810098*z2[i]*r4[i] - 0.317846011338142*r6[i];
      /*c64Re*/  preCoef[totalAN*39+i] = 0.582621362518731*x[i]*z[i]*(33.0*z4[i] - 30.0*z2[i]*r2[i] + 5.0*r4[i]);
      /*c64Im*/  preCoef[totalAN*40+i] = 0.460602629757462*(x2[i] - y2[i])*(33.0*z4[i] - 18.0*z2[i]*r2[i] + r4[i]);
      /*c65Re*/  preCoef[totalAN*41+i] = 0.921205259514923*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 8.0*z2[i]);
      /*c65Im*/  preCoef[totalAN*42+i] = -0.504564900728724*(x2[i] + y2[i] - 10.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
      /*c66Re*/  preCoef[totalAN*43+i] = 2.36661916223175*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
      /*c66Im*/  preCoef[totalAN*44+i] = 0.683184105191914*x6[i] - 10.2477615778787*x4[i]*y2[i] + 10.2477615778787*x2[i]*y4[i] - 0.683184105191914*y6[i];
    }
    if (lMax > 6){
      /*c70  */  preCoef[totalAN*45+i] = 0.707162732524596*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
      /*c71Re*/  preCoef[totalAN*46+i] = 5.2919213236038*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
      /*c71Im*/  preCoef[totalAN*47+i] = -0.51891557872026*y[i]*(x2[i] + y2[i] - 12.0*z2[i])*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
      /*c72Re*/  preCoef[totalAN*48+i] = 4.15132462976208*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 10.0*z2[i]);
      /*c72Im*/  preCoef[totalAN*49+i] = 0.156458933862294*y[i]*(3.0*x2[i] - y2[i])*(143.0*z4[i] - 66.0*z2[i]*r2[i] + 3.0*r4[i]);
      /*c73Re*/  preCoef[totalAN*50+i] = 0.442532692444983*x[i]*y[i]*z[i]*(143.0*z4[i] - 110.0*z2[i]*r2[i] + 15.0*r4[i]);
      /*c73Im*/  preCoef[totalAN*51+i] = 0.0903316075825173*y[i]*(429.0*z6[i] - 495.0*z4[i]*r2[i] + 135.0*z2[i]*r4[i] - 5.0*r6[i]);
      /*c74Re*/  preCoef[totalAN*52+i] = 0.0682842769120049*z[i]*(429.0*z6[i] - 693.0*z4[i]*r2[i] + 315.0*z2[i]*r4[i] - 35.0*r6[i]);
      /*c74Im*/  preCoef[totalAN*53+i] = 0.0903316075825173*x[i]*(429.0*z6[i] - 495.0*z4[i]*r2[i] + 135.0*z2[i]*r4[i] - 5.0*r6[i]);
      /*c75Re*/  preCoef[totalAN*54+i] = 0.221266346222491*z[i]*(x2[i] - y2[i])*(143.0*z4[i] - 110.0*z2[i]*r2[i] + 15.0*r4[i]);
      /*c75Im*/  preCoef[totalAN*55+i] = 0.156458933862294*x[i]*(x2[i] - 3.0*y2[i])*(143.0*z4[i] - 66.0*z2[i]*r2[i] + 3.0*r4[i]);
      /*c76Re*/  preCoef[totalAN*56+i] = 1.03783115744052*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 10.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
      /*c76Im*/  preCoef[totalAN*57+i] = -0.51891557872026*x[i]*(x2[i] + y2[i] - 12.0*z2[i])*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
      /*c77Re*/  preCoef[totalAN*58+i] = 2.6459606618019*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
      /*c77Im*/  preCoef[totalAN*59+i] = 0.707162732524596*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
    }
    if (lMax > 7){
      /*c80  */  preCoef[totalAN*60+i] = 5.83141328139864*x[i]*y[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
      /*c81Re*/  preCoef[totalAN*61+i] = 2.91570664069932*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
      /*c81Im*/  preCoef[totalAN*62+i] = -1.06466553211909*x[i]*y[i]*(x2[i] + y2[i] - 14.0*z2[i])*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
      /*c82Re*/  preCoef[totalAN*63+i] = 3.44991062209811*y[i]*z[i]*(-x2[i] - y2[i] + 4.0*z2[i])*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
      /*c82Im*/  preCoef[totalAN*64+i] = 1.91366609903732*x[i]*y[i]*(x2[i] - y2[i])*(65.0*z4[i] - 26.0*z2[i]*r2[i] + r4[i]);
      /*c83Re*/  preCoef[totalAN*65+i] = 1.23526615529554*y[i]*z[i]*(3.0*x2[i] - y2[i])*(39.0*z4[i] - 26.0*z2[i]*r2[i] + 3.0*r4[i]);
      /*c83Im*/  preCoef[totalAN*66+i] = 0.912304516869819*x[i]*y[i]*(143.0*z6[i] - 143.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i]);
      /*c84Re*/  preCoef[totalAN*67+i] = 0.10904124589878*y[i]*z[i]*(715.0*z6[i] - 1001.0*z4[i]*r2[i] + 385.0*z2[i]*r4[i] - 35.0*r6[i]);
      /*c84Im*/  preCoef[totalAN*68+i] = 58.4733681132208*z8[i] - 109.150287144679*z6[i]*r2[i] + 62.9713195065454*z4[i]*r4[i] - 11.4493308193719*z2[i]*r6[i] + 0.318036967204775*r8[i];
      /*c85Re*/  preCoef[totalAN*69+i] = 0.10904124589878*x[i]*z[i]*(715.0*z6[i] - 1001.0*z4[i]*r2[i] + 385.0*z2[i]*r4[i] - 35.0*r6[i]);
      /*c85Im*/  preCoef[totalAN*70+i] = 0.456152258434909*(x2[i] - y2[i])*(143.0*z6[i] - 143.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i]);
      /*c86Re*/  preCoef[totalAN*71+i] = 1.23526615529554*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(39.0*z4[i] - 26.0*z2[i]*r2[i] + 3.0*r4[i]);
      /*c86Im*/  preCoef[totalAN*72+i] = 0.478416524759331*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(65.0*z4[i] - 26.0*z2[i]*r2[i] + r4[i]);
      /*c87Re*/  preCoef[totalAN*73+i] = 3.44991062209811*x[i]*z[i]*(-x2[i] - y2[i] + 4.0*z2[i])*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
      /*c87Im*/  preCoef[totalAN*74+i] = -0.532332766059543*(x2[i] + y2[i] - 14.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
      /*c88Re*/  preCoef[totalAN*75+i] = 2.91570664069932*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
      /*c88Im*/  preCoef[totalAN*76+i] = 0.72892666017483*x8[i] - 20.4099464848952*x6[i]*y2[i] + 51.0248662122381*x4[i]*y4[i] - 20.4099464848952*x2[i]*y6[i] + 0.72892666017483*y8[i];
    }
    if (lMax > 8){
      /*c90  */  preCoef[totalAN*77+i] = 0.748900951853188*y[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      /*c91Re*/  preCoef[totalAN*78+i] = 25.4185411916376*x[i]*y[i]*z[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
      /*c91Im*/  preCoef[totalAN*79+i] = -0.544905481344053*y[i]*(x2[i] + y2[i] - 16.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
      /*c92Re*/  preCoef[totalAN*80+i] = 2.51681061069513*x[i]*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 14.0*z2[i])*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
      /*c92Im*/  preCoef[totalAN*81+i] = 0.487378279039019*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(85.0*z4[i] - 30.0*z2[i]*r2[i] + r4[i]);
      /*c93Re*/  preCoef[totalAN*82+i] = 16.3107969549167*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(17.0*z4[i] - 10.0*z2[i]*r2[i] + r4[i]);
      /*c93Im*/  preCoef[totalAN*83+i] = 0.461708520016194*y[i]*(3.0*x2[i] - y2[i])*(221.0*z6[i] - 195.0*z4[i]*r2[i] + 39.0*z2[i]*r4[i] - r6[i]);
      /*c94Re*/  preCoef[totalAN*84+i] = 1.209036709703*x[i]*y[i]*z[i]*(221.0*z6[i] - 273.0*z4[i]*r2[i] + 91.0*z2[i]*r4[i] - 7.0*r6[i]);
      /*c94Im*/  preCoef[totalAN*85+i] = 0.0644418731522273*y[i]*(2431.0*z8[i] - 4004.0*z6[i]*r2[i] + 2002.0*z4[i]*r4[i] - 308.0*z2[i]*r6[i] + 7.0*r8[i]);
      /*c95Re*/  preCoef[totalAN*86+i] = 0.00960642726438659*z[i]*(12155.0*z8[i] - 25740.0*z6[i]*r2[i] + 18018.0*z4[i]*r4[i] - 4620.0*z2[i]*r6[i] + 315.0*r8[i]);
      /*c95Im*/  preCoef[totalAN*87+i] = 0.0644418731522273*x[i]*(2431.0*z8[i] - 4004.0*z6[i]*r2[i] + 2002.0*z4[i]*r4[i] - 308.0*z2[i]*r6[i] + 7.0*r8[i]);
      /*c96Re*/  preCoef[totalAN*88+i] = 0.604518354851498*z[i]*(x2[i] - y2[i])*(221.0*z6[i] - 273.0*z4[i]*r2[i] + 91.0*z2[i]*r4[i] - 7.0*r6[i]);
      /*c96Im*/  preCoef[totalAN*89+i] = 0.461708520016194*x[i]*(x2[i] - 3.0*y2[i])*(221.0*z6[i] - 195.0*z4[i]*r2[i] + 39.0*z2[i]*r4[i] - r6[i]);
      /*c97Re*/  preCoef[totalAN*90+i] = 4.07769923872917*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(17.0*z4[i] - 10.0*z2[i]*r2[i] + r4[i]);
      /*c97Im*/  preCoef[totalAN*91+i] = 0.487378279039019*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(85.0*z4[i] - 30.0*z2[i]*r2[i] + r4[i]);
      /*c98Re*/  preCoef[totalAN*92+i] = 1.25840530534757*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 14.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
      /*c98Im*/  preCoef[totalAN*93+i] = -0.544905481344053*x[i]*(x2[i] + y2[i] - 16.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
      /*c99Re*/  preCoef[totalAN*94+i] = 3.1773176489547*z[i]*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      /*c99Im*/  preCoef[totalAN*95+i] = 0.748900951853188*x[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
    }
    if (lMax > 9){ //OBS!!!!! lMax > 9, tesseral cases
      /*l=10*/  preCoef[totalAN*96+i] = 1.53479023644398*x[i]*y[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      /**/  preCoef[totalAN*97+i] = 3.43189529989171*y[i]*z[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      /**/  preCoef[totalAN*98+i] = -4.45381546176335*x[i]*y[i]*(x2[i] + y2[i] - 18.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
      /**/  preCoef[totalAN*99+i] = 1.36369691122981*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 16.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
      /**/  preCoef[totalAN*100+i] = 0.330745082725238*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i]);
      /**/  preCoef[totalAN*101+i] = 0.295827395278969*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(323.0*z4[i] - 170.0*z2[i]*r2[i] + 15.0*r4[i]);
      /**/  preCoef[totalAN*102+i] = 1.87097672671297*x[i]*y[i]*(x2[i] - y2[i])*(323.0*z6[i] - 255.0*z4[i]*r2[i] + 45.0*z2[i]*r4[i] - r6[i]);
      /**/  preCoef[totalAN*103+i] = 0.661490165450475*y[i]*z[i]*(3.0*x2[i] - y2[i])*(323.0*z6[i] - 357.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 7.0*r6[i]);
      /**/  preCoef[totalAN*104+i] = 0.129728894680065*x[i]*y[i]*(4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]);
      /**/  preCoef[totalAN*105+i] = 0.0748990122652082*y[i]*z[i]*(4199.0*z8[i] - 7956.0*z6[i]*r2[i] + 4914.0*z4[i]*r4[i] - 1092.0*z2[i]*r6[i] + 63.0*r8[i]);
      /**/  preCoef[totalAN*106+i] = 233.240148813258*z10[i] - 552.410878768242*z8[i]*r2[i] + 454.926606044435*z6[i]*r4[i] - 151.642202014812*z4[i]*r6[i] + 17.4971771555552*z2[i]*r8[i] - 0.318130493737367*r10[i];
      /**/  preCoef[totalAN*107+i] = 0.0748990122652082*x[i]*z[i]*(4199.0*z8[i] - 7956.0*z6[i]*r2[i] + 4914.0*z4[i]*r4[i] - 1092.0*z2[i]*r6[i] + 63.0*r8[i]);
      /**/  preCoef[totalAN*108+i] = 0.0648644473400325*(x2[i] - y2[i])*(4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]);
      /**/  preCoef[totalAN*109+i] = 0.661490165450475*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(323.0*z6[i] - 357.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 7.0*r6[i]);
      /**/  preCoef[totalAN*110+i] = 0.467744181678242*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(323.0*z6[i] - 255.0*z4[i]*r2[i] + 45.0*z2[i]*r4[i] - r6[i]);
      /**/  preCoef[totalAN*111+i] = 0.295827395278969*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(323.0*z4[i] - 170.0*z2[i]*r2[i] + 15.0*r4[i]);
      /**/  preCoef[totalAN*112+i] = 0.165372541362619*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
      /**/  preCoef[totalAN*113+i] = 1.36369691122981*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 16.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
      /**/  preCoef[totalAN*114+i] = -0.556726932720418*(x2[i] + y2[i] - 18.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      /**/  preCoef[totalAN*115+i] = 3.43189529989171*x[i]*z[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      /**/  preCoef[totalAN*116+i] = 0.76739511822199*x10[i] - 34.5327803199895*x8[i]*y2[i] + 161.152974826618*x6[i]*y4[i] - 161.152974826618*x4[i]*y6[i] + 34.5327803199895*x2[i]*y8[i] - 0.76739511822199*y10[i];
    }

    if (lMax > 10){ //OBS!!!!! lMax > 9, tesseral cases
      preCoef[totalAN*117+i] = 0.784642105787197*y[i]*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*118+i] = 7.36059539761062*x[i]*y[i]*z[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      preCoef[totalAN*119+i] = -0.567882263783437*y[i]*(x2[i] + y2[i] - 20.0*z2[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      preCoef[totalAN*120+i] = 35.1903768038371*x[i]*y[i]*z[i]*(-x2[i] - y2[i] + 6.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
      preCoef[totalAN*121+i] = 0.504576632477118*y[i]*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
      preCoef[totalAN*122+i] = 0.638244565090152*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(399.0*z4[i] - 190.0*z2[i]*r2[i] + 15.0*r4[i]);
      preCoef[totalAN*123+i] = 0.0947934431913346*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(2261.0*z6[i] - 1615.0*z4[i]*r2[i] + 255.0*z2[i]*r4[i] - 5.0*r6[i]);
      preCoef[totalAN*124+i] = 4.0127980256608*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(323.0*z6[i] - 323.0*z4[i]*r2[i] + 85.0*z2[i]*r4[i] - 5.0*r6[i]);
      preCoef[totalAN*125+i] = 0.457895832784712*y[i]*(3.0*x2[i] - y2[i])*(969.0*z8[i] - 1292.0*z6[i]*r2[i] + 510.0*z4[i]*r4[i] - 60.0*z2[i]*r6[i] + r8[i]);
      preCoef[totalAN*126+i] = 0.489511235746264*x[i]*y[i]*z[i]*(2261.0*z8[i] - 3876.0*z6[i]*r2[i] + 2142.0*z4[i]*r4[i] - 420.0*z2[i]*r6[i] + 21.0*r8[i]);
      preCoef[totalAN*127+i] = 0.0214664877426077*y[i]*(29393.0*z10[i] - 62985.0*z8[i]*r2[i] + 46410.0*z6[i]*r4[i] - 13650.0*z4[i]*r6[i] + 1365.0*z2[i]*r8[i] - 21.0*r10[i]);
      preCoef[totalAN*128+i] = 0.00528468396465431*z[i]*(88179.0*z10[i] - 230945.0*z8[i]*r2[i] + 218790.0*z6[i]*r4[i] - 90090.0*z4[i]*r6[i] + 15015.0*z2[i]*r8[i] - 693.0*r10[i]);
      preCoef[totalAN*129+i] = 0.0214664877426077*x[i]*(29393.0*z10[i] - 62985.0*z8[i]*r2[i] + 46410.0*z6[i]*r4[i] - 13650.0*z4[i]*r6[i] + 1365.0*z2[i]*r8[i] - 21.0*r10[i]);
      preCoef[totalAN*130+i] = 0.244755617873132*z[i]*(x2[i] - y2[i])*(2261.0*z8[i] - 3876.0*z6[i]*r2[i] + 2142.0*z4[i]*r4[i] - 420.0*z2[i]*r6[i] + 21.0*r8[i]);
      preCoef[totalAN*131+i] = 0.457895832784712*x[i]*(x2[i] - 3.0*y2[i])*(969.0*z8[i] - 1292.0*z6[i]*r2[i] + 510.0*z4[i]*r4[i] - 60.0*z2[i]*r6[i] + r8[i]);
      preCoef[totalAN*132+i] = 1.0031995064152*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(323.0*z6[i] - 323.0*z4[i]*r2[i] + 85.0*z2[i]*r4[i] - 5.0*r6[i]);
      preCoef[totalAN*133+i] = 0.0947934431913346*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(2261.0*z6[i] - 1615.0*z4[i]*r2[i] + 255.0*z2[i]*r4[i] - 5.0*r6[i]);
      preCoef[totalAN*134+i] = 0.319122282545076*z[i]*(399.0*z4[i] - 190.0*z2[i]*r2[i] + 15.0*r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
      preCoef[totalAN*135+i] = 0.504576632477118*x[i]*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
      preCoef[totalAN*136+i] = 4.39879710047964*z[i]*(-x2[i] - y2[i] + 6.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      preCoef[totalAN*137+i] = -0.567882263783437*x[i]*(x2[i] + y2[i] - 20.0*z2[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      preCoef[totalAN*138+i] = 3.68029769880531*z[i]*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*139+i] = 0.784642105787197*x[i]*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
    }

    if (lMax > 11){ //OBS!!!!! lMax > 9, tesseral cases
      preCoef[totalAN*140+i] = 3.20328798313589*x[i]*y[i]*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      preCoef[totalAN*141+i] = 3.92321052893598*y[i]*z[i]*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*142+i] = -1.15689166958762*x[i]*y[i]*(x2[i] + y2[i] - 22.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      preCoef[totalAN*143+i] = 1.56643872562221*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 20.0*z2[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      preCoef[totalAN*144+i] = 4.10189944667082*x[i]*y[i]*(161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
      preCoef[totalAN*145+i] = 1.0254748616677*y[i]*z[i]*(161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
      preCoef[totalAN*146+i] = 0.192089041114334*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]);
      preCoef[totalAN*147+i] = 1.07809706940566*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(437.0*z6[i] - 399.0*z4[i]*r2[i] + 95.0*z2[i]*r4[i] - 5.0*r6[i]);
      preCoef[totalAN*148+i] = 0.369784244098711*x[i]*y[i]*(x2[i] - y2[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]);
      preCoef[totalAN*149+i] = 0.12326141469957*y[i]*z[i]*(3.0*x2[i] - y2[i])*(7429.0*z8[i] - 11628.0*z6[i]*r2[i] + 5814.0*z4[i]*r4[i] - 1020.0*z2[i]*r6[i] + 45.0*r8[i]);
      preCoef[totalAN*150+i] = 0.301927570987541*x[i]*y[i]*(7429.0*z10[i] - 14535.0*z8[i]*r2[i] + 9690.0*z6[i]*r4[i] - 2550.0*z4[i]*r6[i] + 225.0*z2[i]*r8[i] - 3.0*r10[i]);
      preCoef[totalAN*151+i] = 0.0243300170174164*y[i]*z[i]*(52003.0*z10[i] - 124355.0*z8[i]*r2[i] + 106590.0*z6[i]*r4[i] - 39270.0*z4[i]*r6[i] + 5775.0*z2[i]*r8[i] - 231.0*r10[i]);
      preCoef[totalAN*152+i] = 931.186918632914*z12[i] - 2672.1015925988*z10[i]*r2[i] + 2862.96599207014*z8[i]*r4[i] - 1406.36925926252*z6[i]*r6[i] + 310.228513072616*z4[i]*r8[i] - 24.8182810458093*z2[i]*r10[i] + 0.318183090330888*r12[i];
      preCoef[totalAN*153+i] = 0.0243300170174164*x[i]*z[i]*(52003.0*z10[i] - 124355.0*z8[i]*r2[i] + 106590.0*z6[i]*r4[i] - 39270.0*z4[i]*r6[i] + 5775.0*z2[i]*r8[i] - 231.0*r10[i]);
      preCoef[totalAN*154+i] = 0.150963785493771*(x2[i] - y2[i])*(7429.0*z10[i] - 14535.0*z8[i]*r2[i] + 9690.0*z6[i]*r4[i] - 2550.0*z4[i]*r6[i] + 225.0*z2[i]*r8[i] - 3.0*r10[i]);
      preCoef[totalAN*155+i] = 0.12326141469957*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(7429.0*z8[i] - 11628.0*z6[i]*r2[i] + 5814.0*z4[i]*r4[i] - 1020.0*z2[i]*r6[i] + 45.0*r8[i]);
      preCoef[totalAN*156+i] = 0.0924460610246778*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]);
      preCoef[totalAN*157+i] = 1.07809706940566*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(437.0*z6[i] - 399.0*z4[i]*r2[i] + 95.0*z2[i]*r4[i] - 5.0*r6[i]);
      preCoef[totalAN*158+i] = 0.0960445205571672*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]);
      preCoef[totalAN*159+i] = 1.0254748616677*x[i]*z[i]*(161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
      preCoef[totalAN*160+i] = 0.512737430833852*(161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      preCoef[totalAN*161+i] = 1.56643872562221*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 20.0*z2[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      preCoef[totalAN*162+i] = -0.57844583479381*(x2[i] + y2[i] - 22.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*163+i] = 3.92321052893598*x[i]*z[i]*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      preCoef[totalAN*164+i] = 0.800821995783972*x12[i] - 52.8542517217421*x10[i]*y2[i] + 396.406887913066*x8[i]*y4[i] - 739.95952410439*x6[i]*y6[i] + 396.406887913066*x4[i]*y8[i] - 52.8542517217421*x2[i]*y10[i] + 0.800821995783972*y12[i];
    }

    if (lMax > 12){ //OBS!!!!! lMax > 9, tesseral cases
      preCoef[totalAN*165+i] = 0.816077118837628*y[i]*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*166+i] = 16.6447726141986*x[i]*y[i]*z[i]*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      preCoef[totalAN*167+i] = -0.588481579340398*y[i]*(x2[i] + y2[i] - 24.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*168+i] = 3.32895452283972*x[i]*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 22.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      preCoef[totalAN*169+i] = 0.173533750438143*y[i]*(575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      preCoef[totalAN*170+i] = 14.560298633254*x[i]*y[i]*z[i]*(115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
      preCoef[totalAN*171+i] = 0.486425436917406*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]);
      preCoef[totalAN*172+i] = 2.30218535466597*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]);
      preCoef[totalAN*173+i] = 0.0933659449850856*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i]);
      preCoef[totalAN*174+i] = 0.528157542646753*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]);
      preCoef[totalAN*175+i] = 0.0506347929757152*y[i]*(3.0*x2[i] - y2[i])*(37145.0*z10[i] - 66861.0*z8[i]*r2[i] + 40698.0*z6[i]*r4[i] - 9690.0*z4[i]*r6[i] + 765.0*z2[i]*r8[i] - 9.0*r10[i]);
      preCoef[totalAN*176+i] = 0.122135716100197*x[i]*y[i]*z[i]*(37145.0*z10[i] - 81719.0*z8[i]*r2[i] + 63954.0*z6[i]*r4[i] - 21318.0*z4[i]*r6[i] + 2805.0*z2[i]*r8[i] - 99.0*r10[i]);
      preCoef[totalAN*177+i] = 0.0136551881840328*y[i]*(185725.0*z12[i] - 490314.0*z10[i]*r2[i] + 479655.0*z8[i]*r4[i] - 213180.0*z6[i]*r6[i] + 42075.0*z4[i]*r8[i] - 2970.0*z2[i]*r10[i] + 33.0*r12[i]);
      preCoef[totalAN*178+i] = 0.00143145267159059*z[i]*(1300075.0*z12[i] - 4056234.0*z10[i]*r2[i] + 4849845.0*z8[i]*r4[i] - 2771340.0*z6[i]*r6[i] + 765765.0*z4[i]*r8[i] - 90090.0*z2[i]*r10[i] + 3003.0*r12[i]);
      preCoef[totalAN*179+i] = 0.0136551881840328*x[i]*(185725.0*z12[i] - 490314.0*z10[i]*r2[i] + 479655.0*z8[i]*r4[i] - 213180.0*z6[i]*r6[i] + 42075.0*z4[i]*r8[i] - 2970.0*z2[i]*r10[i] + 33.0*r12[i]);
      preCoef[totalAN*180+i] = 0.0610678580500984*z[i]*(x2[i] - y2[i])*(37145.0*z10[i] - 81719.0*z8[i]*r2[i] + 63954.0*z6[i]*r4[i] - 21318.0*z4[i]*r6[i] + 2805.0*z2[i]*r8[i] - 99.0*r10[i]);
      preCoef[totalAN*181+i] = 0.0506347929757152*x[i]*(x2[i] - 3.0*y2[i])*(37145.0*z10[i] - 66861.0*z8[i]*r2[i] + 40698.0*z6[i]*r4[i] - 9690.0*z4[i]*r6[i] + 765.0*z2[i]*r8[i] - 9.0*r10[i]);
      preCoef[totalAN*182+i] = 0.132039385661688*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]);
      preCoef[totalAN*183+i] = 0.0933659449850856*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i]);
      preCoef[totalAN*184+i] = 1.15109267733299*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]);
      preCoef[totalAN*185+i] = 0.486425436917406*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]);
      preCoef[totalAN*186+i] = 1.82003732915675*z[i]*(115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      preCoef[totalAN*187+i] = 0.173533750438143*x[i]*(575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      preCoef[totalAN*188+i] = 1.66447726141986*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 22.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*189+i] = -0.588481579340398*x[i]*(x2[i] + y2[i] - 24.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      preCoef[totalAN*190+i] = 4.16119315354964*z[i]*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*191+i] = 0.816077118837628*x[i]*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
    }

    if (lMax > 13){ //OBS!!!!! lMax > 9, tesseral cases
      preCoef[totalAN*192+i] = 1.66104416612905*x[i]*y[i]*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      preCoef[totalAN*193+i] = 4.39470978027212*y[i]*z[i]*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*194+i] = -2.39217700650788*x[i]*y[i]*(x2[i] + y2[i] - 26.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      preCoef[totalAN*195+i] = 5.2817838178514*y[i]*z[i]*(-x2[i] - y2[i] + 8.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*196+i] = 1.05635676357028*x[i]*y[i]*(225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      preCoef[totalAN*197+i] = 1.92863476060198*y[i]*z[i]*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      preCoef[totalAN*198+i] = 3.94023104498585*x[i]*y[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i]);
      preCoef[totalAN*199+i] = 0.873160381394213*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]);
      preCoef[totalAN*200+i] = 0.943121003323134*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]);
      preCoef[totalAN*201+i] = 1.265329604663*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]);
      preCoef[totalAN*202+i] = 1.83593315348819*x[i]*y[i]*(x2[i] - y2[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]);
      preCoef[totalAN*203+i] = 0.0652370439174496*y[i]*z[i]*(3.0*x2[i] - y2[i])*(58995.0*z10[i] - 120175.0*z8[i]*r2[i] + 86526.0*z6[i]*r4[i] - 26334.0*z4[i]*r6[i] + 3135.0*z2[i]*r8[i] - 99.0*r10[i]);
      preCoef[totalAN*204+i] = 0.0274050400015396*x[i]*y[i]*(334305.0*z12[i] - 817190.0*z10[i]*r2[i] + 735471.0*z8[i]*r4[i] - 298452.0*z6[i]*r6[i] + 53295.0*z4[i]*r8[i] - 3366.0*z2[i]*r10[i] + 33.0*r12[i]);
      preCoef[totalAN*205+i] = 0.0152015810664143*y[i]*z[i]*(334305.0*z12[i] - 965770.0*z10[i]*r2[i] + 1062347.0*z8[i]*r4[i] - 554268.0*z6[i]*r6[i] + 138567.0*z4[i]*r8[i] - 14586.0*z2[i]*r10[i] + 429.0*r12[i]);
      preCoef[totalAN*206+i] = 3719.61718745389*z14[i] - 12536.487557715*z12[i]*r2[i] + 16548.1635761838*z10[i]*r4[i] - 10792.2805931633*z8[i]*r6[i] + 3597.42686438778*z6[i]*r8[i] - 568.014768061228*z4[i]*r10[i] + 33.4126334153663*z2[i]*r12[i] - 0.318215556336822*r14[i];
      preCoef[totalAN*207+i] = 0.0152015810664143*x[i]*z[i]*(334305.0*z12[i] - 965770.0*z10[i]*r2[i] + 1062347.0*z8[i]*r4[i] - 554268.0*z6[i]*r6[i] + 138567.0*z4[i]*r8[i] - 14586.0*z2[i]*r10[i] + 429.0*r12[i]);
      preCoef[totalAN*208+i] = 0.0137025200007698*(x2[i] - y2[i])*(334305.0*z12[i] - 817190.0*z10[i]*r2[i] + 735471.0*z8[i]*r4[i] - 298452.0*z6[i]*r6[i] + 53295.0*z4[i]*r8[i] - 3366.0*z2[i]*r10[i] + 33.0*r12[i]);
      preCoef[totalAN*209+i] = 0.0652370439174496*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(58995.0*z10[i] - 120175.0*z8[i]*r2[i] + 86526.0*z6[i]*r4[i] - 26334.0*z4[i]*r6[i] + 3135.0*z2[i]*r8[i] - 99.0*r10[i]);
      preCoef[totalAN*210+i] = 0.458983288372048*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]);
      preCoef[totalAN*211+i] = 1.265329604663*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]);
      preCoef[totalAN*212+i] = 0.471560501661567*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]);
      preCoef[totalAN*213+i] = 0.873160381394213*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]);
      preCoef[totalAN*214+i] = 0.492528880623231*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      preCoef[totalAN*215+i] = 1.92863476060198*x[i]*z[i]*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      preCoef[totalAN*216+i] = 0.52817838178514*(225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*217+i] = 5.2817838178514*x[i]*z[i]*(-x2[i] - y2[i] + 8.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      preCoef[totalAN*218+i] = -0.598044251626971*(x2[i] + y2[i] - 26.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*219+i] = 4.39470978027212*x[i]*z[i]*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      preCoef[totalAN*220+i] = 0.830522083064524*x14[i] - 75.5775095588717*x12[i]*y2[i] + 831.352605147589*x10[i]*y4[i] - 2494.05781544277*x8[i]*y6[i] + 2494.05781544277*x6[i]*y8[i] - 831.352605147589*x4[i]*y10[i] + 75.5775095588717*x2[i]*y12[i] - 0.830522083064524*y14[i];
    }

    if (lMax > 14){ //OBS!!!!! lMax > 9, tesseral cases
      preCoef[totalAN*221+i] = 0.844250650857373*y[i]*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*222+i] = 9.24830251326002*x[i]*y[i]*z[i]*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      preCoef[totalAN*223+i] = -0.60718080651189*y[i]*(x2[i] + y2[i] - 28.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*224+i] = 7.41987201697353*x[i]*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 26.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      preCoef[totalAN*225+i] = 0.53548313829403*y[i]*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*226+i] = 2.44217885935126*x[i]*y[i]*z[i]*(261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      preCoef[totalAN*227+i] = 0.49850767216857*y[i]*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      preCoef[totalAN*228+i] = 7.38445476455181*x[i]*y[i]*z[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i]);
      preCoef[totalAN*229+i] = 0.0680486534764293*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]);
      preCoef[totalAN*230+i] = 0.638352953401215*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]);
      preCoef[totalAN*231+i] = 0.462530657238986*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]);
      preCoef[totalAN*232+i] = 2.49470484396446*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]);
      preCoef[totalAN*233+i] = 0.0413039660894728*y[i]*(3.0*x2[i] - y2[i])*(190095.0*z12[i] - 432630.0*z10[i]*r2[i] + 360525.0*z8[i]*r4[i] - 134596.0*z6[i]*r6[i] + 21945.0*z4[i]*r8[i] - 1254.0*z2[i]*r10[i] + 11.0*r12[i]);
      preCoef[totalAN*234+i] = 0.0324014967813841*x[i]*y[i]*z[i]*(570285.0*z12[i] - 1533870.0*z10[i]*r2[i] + 1562275.0*z8[i]*r4[i] - 749892.0*z6[i]*r6[i] + 171171.0*z4[i]*r8[i] - 16302.0*z2[i]*r10[i] + 429.0*r12[i]);
      preCoef[totalAN*235+i] = 0.00105013854311783*y[i]*(9694845.0*z14[i] - 30421755.0*z12[i]*r2[i] + 37182145.0*z10[i]*r4[i] - 22309287.0*z8[i]*r6[i] + 6789783.0*z6[i]*r8[i] - 969969.0*z4[i]*r10[i] + 51051.0*z2[i]*r12[i] - 429.0*r14[i]);
      preCoef[totalAN*236+i] = 0.000766912758094997*z[i]*(9694845.0*z14[i] - 35102025.0*z12[i]*r2[i] + 50702925.0*z10[i]*r4[i] - 37182145.0*z8[i]*r6[i] + 14549535.0*z6[i]*r8[i] - 2909907.0*z4[i]*r10[i] + 255255.0*z2[i]*r12[i] - 6435.0*r14[i]);
      preCoef[totalAN*237+i] = 0.00105013854311783*x[i]*(9694845.0*z14[i] - 30421755.0*z12[i]*r2[i] + 37182145.0*z10[i]*r4[i] - 22309287.0*z8[i]*r6[i] + 6789783.0*z6[i]*r8[i] - 969969.0*z4[i]*r10[i] + 51051.0*z2[i]*r12[i] - 429.0*r14[i]);
      preCoef[totalAN*238+i] = 0.016200748390692*z[i]*(x2[i] - y2[i])*(570285.0*z12[i] - 1533870.0*z10[i]*r2[i] + 1562275.0*z8[i]*r4[i] - 749892.0*z6[i]*r6[i] + 171171.0*z4[i]*r8[i] - 16302.0*z2[i]*r10[i] + 429.0*r12[i]);
      preCoef[totalAN*239+i] = 0.0413039660894728*x[i]*(x2[i] - 3.0*y2[i])*(190095.0*z12[i] - 432630.0*z10[i]*r2[i] + 360525.0*z8[i]*r4[i] - 134596.0*z6[i]*r6[i] + 21945.0*z4[i]*r8[i] - 1254.0*z2[i]*r10[i] + 11.0*r12[i]);
      preCoef[totalAN*240+i] = 0.623676210991114*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]);
      preCoef[totalAN*241+i] = 0.462530657238986*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]);
      preCoef[totalAN*242+i] = 0.319176476700607*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]);
      preCoef[totalAN*243+i] = 0.0680486534764293*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]);
      preCoef[totalAN*244+i] = 0.923056845568976*z[i]*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      preCoef[totalAN*245+i] = 0.49850767216857*x[i]*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      preCoef[totalAN*246+i] = 1.22108942967563*z[i]*(261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*247+i] = 0.53548313829403*x[i]*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      preCoef[totalAN*248+i] = 1.85496800424338*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 26.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*249+i] = -0.60718080651189*x[i]*(x2[i] + y2[i] - 28.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      preCoef[totalAN*250+i] = 4.62415125663001*z[i]*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*251+i] = 0.844250650857373*x[i]*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
    }

    if (lMax > 15){ //OBS!!!!! lMax > 9, tesseral cases
      preCoef[totalAN*252+i] = 13.7174494214084*x[i]*y[i]*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*253+i] = 4.84985075323068*y[i]*z[i]*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*254+i] = -1.23186332318453*x[i]*y[i]*(x2[i] + y2[i] - 30.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      preCoef[totalAN*255+i] = 1.94774693364361*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 28.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*256+i] = 0.723375051052533*x[i]*y[i]*(899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      preCoef[totalAN*257+i] = 0.427954451513054*y[i]*z[i]*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*258+i] = 0.2017396631359*x[i]*y[i]*(8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      preCoef[totalAN*259+i] = 0.194401203675805*y[i]*z[i]*(8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      preCoef[totalAN*260+i] = 0.549849637559955*x[i]*y[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]);
      preCoef[totalAN*261+i] = 0.336712761819039*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]);
      preCoef[totalAN*262+i] = 0.0444043640573282*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]);
      preCoef[totalAN*263+i] = 0.031398626939213*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]);
      preCoef[totalAN*264+i] = 0.166145916780101*x[i]*y[i]*(x2[i] - y2[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]);
      preCoef[totalAN*265+i] = 0.0515196617272515*y[i]*z[i]*(3.0*x2[i] - y2[i])*(310155.0*z12[i] - 780390.0*z10[i]*r2[i] + 740025.0*z8[i]*r4[i] - 328900.0*z6[i]*r6[i] + 69069.0*z4[i]*r8[i] - 6006.0*z2[i]*r10[i] + 143.0*r12[i]);
      preCoef[totalAN*266+i] = 0.00631774627238717*x[i]*y[i]*(5892945.0*z14[i] - 17298645.0*z12[i]*r2[i] + 19684665.0*z10[i]*r4[i] - 10935925.0*z8[i]*r6[i] + 3062059.0*z6[i]*r8[i] - 399399.0*z4[i]*r10[i] + 19019.0*z2[i]*r12[i] - 143.0*r14[i]);
      preCoef[totalAN*267+i] = 0.00115345738199354*y[i]*z[i]*(17678835.0*z14[i] - 59879925.0*z12[i]*r2[i] + 80528175.0*z10[i]*r4[i] - 54679625.0*z8[i]*r6[i] + 19684665.0*z6[i]*r8[i] - 3594591.0*z4[i]*r10[i] + 285285.0*z2[i]*r12[i] - 6435.0*r14[i]);
      preCoef[totalAN*268+i] = 14862.9380228203*z16[i] - 57533.9536367237*z14[i]*r2[i] + 90268.7893265838*z12[i]*r4[i] - 73552.3468586979*z10[i]*r6[i] + 33098.5560864141*z8[i]*r8[i] - 8058.77887321386*z6[i]*r10[i] + 959.378437287364*z4[i]*r12[i] - 43.2802302535653*z2[i]*r14[i] + 0.318236987158568*r16[i];
      preCoef[totalAN*269+i] = 0.00115345738199354*x[i]*z[i]*(17678835.0*z14[i] - 59879925.0*z12[i]*r2[i] + 80528175.0*z10[i]*r4[i] - 54679625.0*z8[i]*r6[i] + 19684665.0*z6[i]*r8[i] - 3594591.0*z4[i]*r10[i] + 285285.0*z2[i]*r12[i] - 6435.0*r14[i]);
      preCoef[totalAN*270+i] = 0.00315887313619359*(x2[i] - y2[i])*(5892945.0*z14[i] - 17298645.0*z12[i]*r2[i] + 19684665.0*z10[i]*r4[i] - 10935925.0*z8[i]*r6[i] + 3062059.0*z6[i]*r8[i] - 399399.0*z4[i]*r10[i] + 19019.0*z2[i]*r12[i] - 143.0*r14[i]);
      preCoef[totalAN*271+i] = 0.0515196617272515*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(310155.0*z12[i] - 780390.0*z10[i]*r2[i] + 740025.0*z8[i]*r4[i] - 328900.0*z6[i]*r6[i] + 69069.0*z4[i]*r8[i] - 6006.0*z2[i]*r10[i] + 143.0*r12[i]);
      preCoef[totalAN*272+i] = 0.0415364791950254*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]);
      preCoef[totalAN*273+i] = 0.031398626939213*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]);
      preCoef[totalAN*274+i] = 0.0222021820286641*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]);
      preCoef[totalAN*275+i] = 0.336712761819039*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]);
      preCoef[totalAN*276+i] = 0.0687312046949944*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]);
      preCoef[totalAN*277+i] = 0.194401203675805*x[i]*z[i]*(8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      preCoef[totalAN*278+i] = 0.10086983156795*(8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*279+i] = 0.427954451513054*x[i]*z[i]*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      preCoef[totalAN*280+i] = 0.180843762763133*(899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*281+i] = 1.94774693364361*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 28.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      preCoef[totalAN*282+i] = -0.615931661592266*(x2[i] + y2[i] - 30.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*283+i] = 4.84985075323068*x[i]*z[i]*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
      preCoef[totalAN*284+i] = 0.857340588838025*x16[i] - 102.880870660563*x14[i]*y2[i] + 1560.35987168521*x12[i]*y4[i] - 6865.5834354149*x10[i]*y6[i] + 11033.9733783454*x8[i]*y8[i] - 6865.5834354149*x6[i]*y10[i] + 1560.35987168521*x4[i]*y12[i] - 102.880870660563*x2[i]*y14[i] + 0.857340588838025*y16[i];
    }

    if (lMax > 16){ //OBS!!!!! lMax > 9, tesseral cases
      preCoef[totalAN*285+i] = 0.869857171920628*y[i]*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]);
      preCoef[totalAN*286+i] = 81.1535251976858*x[i]*y[i]*z[i]*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*287+i] = -0.624331775925749*y[i]*(x2[i] + y2[i] - 32.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*288+i] = 12.2343542497898*x[i]*y[i]*z[i]*(-x2[i] - y2[i] + 10.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      preCoef[totalAN*289+i] = 0.549338722534015*y[i]*(341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*290+i] = 1.79413275488091*x[i]*y[i]*z[i]*(1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      preCoef[totalAN*291+i] = 0.102009639854704*y[i]*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*292+i] = 0.408038559418814*x[i]*y[i]*z[i]*(9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      preCoef[totalAN*293+i] = 0.013881753693839*y[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]);
      preCoef[totalAN*294+i] = 1.69879999122657*x[i]*y[i]*z[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]);
      preCoef[totalAN*295+i] = 0.0671509657668754*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]);
      preCoef[totalAN*296+i] = 0.727382699731321*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]);
      preCoef[totalAN*297+i] = 0.0656749401202387*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]);
      preCoef[totalAN*298+i] = 0.0310675249591503*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]);
      preCoef[totalAN*299+i] = 0.00317081598837913*y[i]*(3.0*x2[i] - y2[i])*(10235115.0*z14[i] - 28224105.0*z12[i]*r2[i] + 30045015.0*z10[i]*r4[i] - 15540525.0*z8[i]*r6[i] + 4029025.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 21021.0*z2[i]*r12[i] - 143.0*r14[i]);
      preCoef[totalAN*300+i] = 0.0219680575732975*x[i]*y[i]*z[i]*(3411705.0*z14[i] - 10855425.0*z12[i]*r2[i] + 13656825.0*z10[i]*r4[i] - 8633625.0*z8[i]*r6[i] + 2877875.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 35035.0*z2[i]*r12[i] - 715.0*r14[i]);
      preCoef[totalAN*301+i] = 0.0006299772562361*y[i]*(64822395.0*z16[i] - 235717800.0*z14[i]*r2[i] + 345972900.0*z12[i]*r4[i] - 262462200.0*z10[i]*r6[i] + 109359250.0*z8[i]*r8[i] - 24496472.0*z6[i]*r10[i] + 2662660.0*z4[i]*r12[i] - 108680.0*z2[i]*r14[i] + 715.0*r16[i]);
      preCoef[totalAN*302+i] = 5.09306425332988e-5*z[i]*(583401555.0*z16[i] - 2404321560.0*z14[i]*r2[i] + 4071834900.0*z12[i]*r4[i] - 3650610600.0*z10[i]*r6[i] + 1859107250.0*z8[i]*r8[i] - 535422888.0*z6[i]*r10[i] + 81477396.0*z4[i]*r12[i] - 5542680.0*z2[i]*r14[i] + 109395.0*r16[i]);
      preCoef[totalAN*303+i] = 0.0006299772562361*x[i]*(64822395.0*z16[i] - 235717800.0*z14[i]*r2[i] + 345972900.0*z12[i]*r4[i] - 262462200.0*z10[i]*r6[i] + 109359250.0*z8[i]*r8[i] - 24496472.0*z6[i]*r10[i] + 2662660.0*z4[i]*r12[i] - 108680.0*z2[i]*r14[i] + 715.0*r16[i]);
      preCoef[totalAN*304+i] = 0.0109840287866487*z[i]*(x2[i] - y2[i])*(3411705.0*z14[i] - 10855425.0*z12[i]*r2[i] + 13656825.0*z10[i]*r4[i] - 8633625.0*z8[i]*r6[i] + 2877875.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 35035.0*z2[i]*r12[i] - 715.0*r14[i]);
      preCoef[totalAN*305+i] = 0.00317081598837913*x[i]*(x2[i] - 3.0*y2[i])*(10235115.0*z14[i] - 28224105.0*z12[i]*r2[i] + 30045015.0*z10[i]*r4[i] - 15540525.0*z8[i]*r6[i] + 4029025.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 21021.0*z2[i]*r12[i] - 143.0*r14[i]);
      preCoef[totalAN*306+i] = 0.00776688123978757*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]);
      preCoef[totalAN*307+i] = 	0.0656749401202387*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]);
      preCoef[totalAN*308+i] = 0.36369134986566*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]);
      preCoef[totalAN*309+i] = 0.0671509657668754*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]);
      preCoef[totalAN*310+i] = 0.212349998903322*z[i]*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]);
      preCoef[totalAN*311+i] = 0.013881753693839*x[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]);
      preCoef[totalAN*312+i] = 0.204019279709407*z[i]*(9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*313+i] = 0.102009639854704*x[i]*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      preCoef[totalAN*314+i] = 0.448533188720228*z[i]*(1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*315+i] = 0.549338722534015*x[i]*(341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      preCoef[totalAN*316+i] = 6.11717712489491*z[i]*(-x2[i] - y2[i] + 10.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*317+i] = -0.624331775925749*x[i]*(x2[i] + y2[i] - 32.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
      preCoef[totalAN*318+i] = 5.07209532485536*z[i]*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]);
      preCoef[totalAN*319+i] = 0.869857171920628*x[i]*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]);
    }

    if (lMax > 17){ //OBS!!!!! lMax > 9, tesseral cases
      preCoef[totalAN*320+i] = 1.76371153735666*x[i]*y[i]*(9.0*x16[i] - 408.0*x14[i]*y2[i] + 4284.0*x12[i]*y4[i] - 15912.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 15912.0*x6[i]*y10[i] + 4284.0*x4[i]*y12[i] - 408.0*x2[i]*y14[i] + 9.0*y16[i]);
      preCoef[totalAN*321+i] = 5.29113461206997*y[i]*z[i]*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]);
      preCoef[totalAN*322+i] = -10.1185847426968*x[i]*y[i]*(x2[i] + y2[i] - 34.0*z2[i])*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*323+i] = 2.12901451204377*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 32.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*324+i] = 1.11184156725673*x[i]*y[i]*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      preCoef[totalAN*325+i] = 7.03190349956512*y[i]*z[i]*(77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*326+i] = 2.06241672263956*x[i]*y[i]*(2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      preCoef[totalAN*327+i] = 1.49436288677056*y[i]*z[i]*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*328+i] = 0.1962194600598*x[i]*y[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i])*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i]);
      preCoef[totalAN*329+i] = 0.0247213282718858*y[i]*z[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]);
      preCoef[totalAN*330+i] = 0.541617165840082*x[i]*y[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]);
      preCoef[totalAN*331+i] = 1.14494717494913*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]);
      preCoef[totalAN*332+i] = 0.132207111932957*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]);
      preCoef[totalAN*333+i] = 0.0898170459553624*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]);
      preCoef[totalAN*334+i] = 0.140148631920647*x[i]*y[i]*(x2[i] - y2[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]);
      preCoef[totalAN*335+i] = 0.0192873206845831*y[i]*z[i]*(3.0*x2[i] - y2[i])*(3411705.0*z14[i] - 10235115.0*z12[i]*r2[i] + 12096045.0*z10[i]*r4[i] - 7153575.0*z8[i]*r6[i] + 2220075.0*z6[i]*r8[i] - 345345.0*z4[i]*r10[i] + 23023.0*z2[i]*r12[i] - 429.0*r14[i]);
      preCoef[totalAN*336+i] = 0.0063132576421421*x[i]*y[i]*(23881935.0*z16[i] - 81880920.0*z14[i]*r2[i] + 112896420.0*z12[i]*r4[i] - 80120040.0*z10[i]*r6[i] + 31081050.0*z8[i]*r8[i] - 6446440.0*z6[i]*r10[i] + 644644.0*z4[i]*r12[i] - 24024.0*z2[i]*r14[i] + 143.0*r16[i]);
      preCoef[totalAN*337+i] = 0.000684768935318508*y[i]*z[i]*(119409675.0*z16[i] - 463991880.0*z14[i]*r2[i] + 738168900.0*z12[i]*r4[i] - 619109400.0*z10[i]*r6[i] + 293543250.0*z8[i]*r8[i] - 78278200.0*z6[i]*r10[i] + 10958948.0*z4[i]*r12[i] - 680680.0*z2[i]*r14[i] + 12155.0*r16[i]);
      preCoef[totalAN*338+i] = 59403.1009679377*z18[i] - 259676.412802699*z16[i]*r2[i] + 472138.932368544*z14[i]*r4[i] - 461985.406941263*z12[i]*r6[i] + 262853.766018305*z10[i]*r8[i] - 87617.9220061016*z8[i]*r10[i] + 16355.345441139*z6[i]*r12[i] - 1523.7899479322*z4[i]*r14[i] + 54.4210695690072*z2[i]*r16[i] - 0.318251868824604*r18[i];
      preCoef[totalAN*339+i] = 0.000684768935318508*x[i]*z[i]*(119409675.0*z16[i] - 463991880.0*z14[i]*r2[i] + 738168900.0*z12[i]*r4[i] - 619109400.0*z10[i]*r6[i] + 293543250.0*z8[i]*r8[i] - 78278200.0*z6[i]*r10[i] + 10958948.0*z4[i]*r12[i] - 680680.0*z2[i]*r14[i] + 12155.0*r16[i]);
      preCoef[totalAN*340+i] = 0.00315662882107105*(x2[i] - y2[i])*(23881935.0*z16[i] - 81880920.0*z14[i]*r2[i] + 112896420.0*z12[i]*r4[i] - 80120040.0*z10[i]*r6[i] + 31081050.0*z8[i]*r8[i] - 6446440.0*z6[i]*r10[i] + 644644.0*z4[i]*r12[i] - 24024.0*z2[i]*r14[i] + 143.0*r16[i]);
      preCoef[totalAN*341+i] = 0.0192873206845831*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(3411705.0*z14[i] - 10235115.0*z12[i]*r2[i] + 12096045.0*z10[i]*r4[i] - 7153575.0*z8[i]*r6[i] + 2220075.0*z6[i]*r8[i] - 345345.0*z4[i]*r10[i] + 23023.0*z2[i]*r12[i] - 429.0*r14[i]);
      preCoef[totalAN*342+i] = 0.0350371579801619*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]);
      preCoef[totalAN*343+i] = 0.0898170459553624*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]);
      preCoef[totalAN*344+i] = 0.0661035559664783*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]);
      preCoef[totalAN*345+i] = 1.14494717494913*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]);
      preCoef[totalAN*346+i] = 0.0677021457300103*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]);
      preCoef[totalAN*347+i] = 0.0247213282718858*x[i]*z[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]);
      preCoef[totalAN*348+i] = 0.0981097300298999*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*349+i] = 1.49436288677056*x[i]*z[i]*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      preCoef[totalAN*350+i] = 0.515604180659891*(2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*351+i] = 7.03190349956512*x[i]*z[i]*(77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      preCoef[totalAN*352+i] = 0.555920783628365*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*353+i] = 2.12901451204377*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 32.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
      preCoef[totalAN*354+i] = -0.632411546418547*(x2[i] + y2[i] - 34.0*z2[i])*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]);
      preCoef[totalAN*355+i] = 5.29113461206997*x[i]*z[i]*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]);
      preCoef[totalAN*356+i] = 0.881855768678329*x18[i] - 134.923932607784*x16[i]*y2[i] + 2698.47865215569*x14[i]*y4[i] - 16370.7704897445*x12[i]*y6[i] + 38588.2447258263*x10[i]*y8[i] - 38588.2447258263*x8[i]*y10[i] + 16370.7704897445*x6[i]*y12[i] - 2698.47865215569*x4[i]*y14[i] + 134.923932607784*x2[i]*y16[i] - 0.881855768678329*y18[i];
    }

    if (lMax > 18){ //OBS!!!!! lMax > 9, tesseral cases
      preCoef[totalAN*357+i] = 0.893383784349949*y[i]*(19.0*x18[i] - 969.0*x16[i]*y2[i] + 11628.0*x14[i]*y4[i] - 50388.0*x12[i]*y6[i] + 92378.0*x10[i]*y8[i] - 75582.0*x8[i]*y10[i] + 27132.0*x6[i]*y12[i] - 3876.0*x4[i]*y14[i] + 171.0*x2[i]*y16[i] - y18[i]);
      preCoef[totalAN*358+i] = 11.0143750205445*x[i]*y[i]*z[i]*(9.0*x16[i] - 408.0*x14[i]*y2[i] + 4284.0*x12[i]*y4[i] - 15912.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 15912.0*x6[i]*y10[i] + 4284.0*x4[i]*y12[i] - 408.0*x2[i]*y14[i] + 9.0*y16[i]);
      preCoef[totalAN*359+i] = -0.640197544188601*y[i]*(x2[i] + y2[i] - 36.0*z2[i])*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]);
      preCoef[totalAN*360+i] = 35.4833495492953*x[i]*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 34.0*z2[i])*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*361+i] = 0.187430649022538*y[i]*(1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*362+i] = 4.8875933516657*x[i]*y[i]*z[i]*(259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      preCoef[totalAN*363+i] = 0.521019201915023*y[i]*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*364+i] = 31.1916055279426*x[i]*y[i]*z[i]*(407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      preCoef[totalAN*365+i] = 0.495167232923569*y[i]*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*366+i] = 0.361619017612871*x[i]*y[i]*z[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i])*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i]);
      preCoef[totalAN*367+i] = 0.0530874997258304*y[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i]);
      preCoef[totalAN*368+i] = 1.06477924459395*x[i]*y[i]*z[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i]);
      preCoef[totalAN*369+i] = 0.0665487027871216*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i]);
      preCoef[totalAN*370+i] = 0.188228156079767*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i]);
      preCoef[totalAN*371+i] = 0.0352142635297349*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i]);
      preCoef[totalAN*372+i] = 0.890858231034903*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]);
      preCoef[totalAN*373+i] = 0.0116097988782603*y[i]*(3.0*x2[i] - y2[i])*(11475735.0*z16[i] - 37218600.0*z14[i]*r2[i] + 48384180.0*z12[i]*r4[i] - 32256120.0*z10[i]*r6[i] + 11705850.0*z8[i]*r8[i] - 2260440.0*z6[i]*r10[i] + 209300.0*z4[i]*r12[i] - 7176.0*z2[i]*r14[i] + 39.0*r16[i]);
      preCoef[totalAN*374+i] = 0.00240131363330655*x[i]*y[i]*z[i]*(126233085.0*z16[i] - 463991880.0*z14[i]*r2[i] + 695987820.0*z12[i]*r4[i] - 548354040.0*z10[i]*r6[i] + 243221550.0*z8[i]*r8[i] - 60386040.0*z6[i]*r10[i] + 7827820.0*z4[i]*r12[i] - 447304.0*z2[i]*r14[i] + 7293.0*r16[i]);
      preCoef[totalAN*375+i] = 0.000185265368964422*y[i]*(883631595.0*z18[i] - 3653936055.0*z16[i]*r2[i] + 6263890380.0*z14[i]*r4[i] - 5757717420.0*z12[i]*r6[i] + 3064591530.0*z10[i]*r8[i] - 951080130.0*z8[i]*r10[i] + 164384220.0*z6[i]*r12[i] - 14090076.0*z4[i]*r14[i] + 459459.0*z2[i]*r16[i] - 2431.0*r18[i]);
      preCoef[totalAN*376+i] = 2.68811250303113e-5*z[i]*(4418157975.0*z18[i] - 20419054425.0*z16[i]*r2[i] + 39671305740.0*z14[i]*r4[i] - 42075627300.0*z12[i]*r6[i] + 26466926850.0*z10[i]*r8[i] - 10039179150.0*z8[i]*r10[i] + 2230928700.0*z6[i]*r12[i] - 267711444.0*z4[i]*r14[i] + 14549535.0*z2[i]*r16[i] - 230945.0*r18[i]);
      preCoef[totalAN*377+i] = 0.000185265368964422*x[i]*(883631595.0*z18[i] - 3653936055.0*z16[i]*r2[i] + 6263890380.0*z14[i]*r4[i] - 5757717420.0*z12[i]*r6[i] + 3064591530.0*z10[i]*r8[i] - 951080130.0*z8[i]*r10[i] + 164384220.0*z6[i]*r12[i] - 14090076.0*z4[i]*r14[i] + 459459.0*z2[i]*r16[i] - 2431.0*r18[i]);
      preCoef[totalAN*378+i] = 0.00120065681665328*z[i]*(x2[i] - y2[i])*(126233085.0*z16[i] - 463991880.0*z14[i]*r2[i] + 695987820.0*z12[i]*r4[i] - 548354040.0*z10[i]*r6[i] + 243221550.0*z8[i]*r8[i] - 60386040.0*z6[i]*r10[i] + 7827820.0*z4[i]*r12[i] - 447304.0*z2[i]*r14[i] + 7293.0*r16[i]);
      preCoef[totalAN*379+i] = 0.0116097988782603*x[i]*(x2[i] - 3.0*y2[i])*(11475735.0*z16[i] - 37218600.0*z14[i]*r2[i] + 48384180.0*z12[i]*r4[i] - 32256120.0*z10[i]*r6[i] + 11705850.0*z8[i]*r8[i] - 2260440.0*z6[i]*r10[i] + 209300.0*z4[i]*r12[i] - 7176.0*z2[i]*r14[i] + 39.0*r16[i]);
      preCoef[totalAN*380+i] = 0.222714557758726*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]);
      preCoef[totalAN*381+i] = 0.0352142635297349*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i]);
      preCoef[totalAN*382+i] = 0.0941140780398835*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i]);
      preCoef[totalAN*383+i] = 0.0665487027871216*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i]);
      preCoef[totalAN*384+i] = 0.133097405574243*z[i]*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i]);
      preCoef[totalAN*385+i] = 0.0530874997258304*x[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i]);
      preCoef[totalAN*386+i] = 0.180809508806436*z[i]*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      preCoef[totalAN*387+i] = 0.495167232923569*x[i]*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      preCoef[totalAN*388+i] = 7.79790138198564*z[i]*(407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      preCoef[totalAN*389+i] = 0.521019201915023*x[i]*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      preCoef[totalAN*390+i] = 2.44379667583285*z[i]*(259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      preCoef[totalAN*391+i] = 0.187430649022538*x[i]*(1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
      preCoef[totalAN*392+i] = 2.21770934683096*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 34.0*z2[i])*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]);
      preCoef[totalAN*393+i] = -0.640197544188601*x[i]*(x2[i] + y2[i] - 36.0*z2[i])*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]);
      preCoef[totalAN*394+i] = 5.50718751027224*z[i]*(x18[i] - 153.0*x16[i]*y2[i] + 3060.0*x14[i]*y4[i] - 18564.0*x12[i]*y6[i] + 43758.0*x10[i]*y8[i] - 43758.0*x8[i]*y10[i] + 18564.0*x6[i]*y12[i] - 3060.0*x4[i]*y14[i] + 153.0*x2[i]*y16[i] - y18[i]);
      preCoef[totalAN*395+i] = 0.893383784349949*x[i]*(x18[i] - 171.0*x16[i]*y2[i] + 3876.0*x14[i]*y4[i] - 27132.0*x12[i]*y6[i] + 75582.0*x10[i]*y8[i] - 92378.0*x8[i]*y10[i] + 50388.0*x6[i]*y12[i] - 11628.0*x4[i]*y14[i] + 969.0*x2[i]*y16[i] - 19.0*y18[i]);
    }


  }
}
//================================================================
void getCD(double* CDev, double* C, double* preCoef,  double* x, double* y, double* z,double* r2, double* bOa, double* aOa, double* exes,  int totalAN, int Asize, int Ns, int Ntypes, int lMax, int posI, int typeJ, const vector<int> &indices){

  if(Asize == 0){return;}
  double sumMe = 0; int NsNs = Ns*Ns;  int NsJ = ((lMax+1)*(lMax+1))*Ns*typeJ; int LNsNs;
  int LNs; int NsTsI = ((lMax+1)*(lMax+1))*Ns*Ntypes*posI;
  //Non dev  l=0
  for(int k = 0; k < Ns; k++){
    sumMe = 0;
    for(int i = 0; i < Asize; i++){ sumMe += exp(aOa[k]*r2[i]);}
    for(int n = 0; n < Ns; n++){
	    C[NsTsI + NsJ + n] += bOa[n*Ns + k]*sumMe;
    }}
    
  //dev l=0
    for(int i = 0; i < Asize; i++){
     for(int k = 0; k < Ns; k++){
       for(int n = 0; n < Ns; n++){
	    CDev[NsTsI*totalAN + NsJ*totalAN + n*totalAN + indices[i]] += bOa[n*Ns + k]*(-2.0)*x[i]*aOa[k]*exp(aOa[k]*r2[i]);
//	    std::cout << CDev[NsTsI*totalAN + NsJ*totalAN + n*totalAN + indices[i]]  << std::endl; 
//	    std::cout << bOa[n*Ns + k]*(-2)*x[i]*aOa[k]*exp(aOa[k]*r2[i])  << std::endl; 
//	    std::cout << bOa[n*Ns + k]*(-2.0)*x[i]*aOa[k]*exp(aOa[k]*r2[i])  << std::endl; 
       }
     }
    }

   if(lMax > 0) { LNsNs=NsNs; LNs=Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c10*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*z[i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c11Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*x[i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*2 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c11Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*y[i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*3 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }} if(lMax > 1) { LNsNs=2*NsNs; LNs=2*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c20*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*preCoef[i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*4 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c21Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*preCoef[totalAN + i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*5 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c21Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*preCoef[totalAN*2+ i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*6 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c22Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*preCoef[totalAN*3+ i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*7 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c22Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*preCoef[totalAN*4+ i];}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*8 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
  }} if(lMax > 2) { LNsNs=3*NsNs; LNs=3*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c30*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*5+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*9 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c31Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*6+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*10 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c31Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*7+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*11 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c32Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*8+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*12 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c32Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*9+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*13 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c33Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*10+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*14 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c33Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*11+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*15 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
  }} if(lMax > 3) { LNsNs=4*NsNs; LNs=4*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c40*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*12+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*16 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c41Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*13+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*17 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c41Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*14+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*18 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c42Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*15+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*19 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c42Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*16+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*20 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c43Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*17+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*21 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c43Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*18+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*22 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c44Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*19+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*23 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }
    sumMe = 0;/*c44Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*20+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*24 + n] += bOa[LNsNs + n*Ns + k]*sumMe; }

  }} if(lMax > 4) { LNsNs=5*NsNs; LNs=5*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c50*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*21+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*25 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c51Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*22+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*26 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c51Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*23+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*27 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c52Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*24+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*28 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c52Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*25+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*29 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c53Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*26+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*30 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c53Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*27+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*31 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c54Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*28+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*32 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c54Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*29+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*33 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c55Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*30+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*34 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c55Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*31+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*35 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }} if(lMax > 5) { LNsNs=6*NsNs; LNs=6*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c60*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*32+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*36 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c61Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*33+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*37 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c61Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*34+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*38 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c62Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*35+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*39 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c62Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*36+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*40 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c63Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*37+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*41 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c63Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*38+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*42 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c64Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*39+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*43 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c64Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*40+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*44 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c65Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*41+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*45 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c65Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*42+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*46 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c66Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*43+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*47 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c66Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*44+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*48 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }} if(lMax > 6) { LNsNs=7*NsNs; LNs=7*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c70*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*45+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*49 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c71Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*46+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*50 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c71Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*47+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*51 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c72Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*48+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*52 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c72Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*49+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*53 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c73Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*50+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*54 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c73Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*51+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*55 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c74Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*52+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*56 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c74Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*53+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*57 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c75Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*54+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*58 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c75Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*55+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*59 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c76Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*56+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*60 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c76Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*57+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*61 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c77Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*58+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*62 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c77Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*59+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*63 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }} if(lMax > 7) { LNsNs=8*NsNs; LNs=8*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c80*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*60+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*64 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c81Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*61+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*65 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c81Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*62+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*66 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c82Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*63+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*67 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c82Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*64+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*68 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c83Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*65+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*69 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c83Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*66+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*70 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c84Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*67+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*71 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c84Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*68+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*72 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c85Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*69+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*73 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c85Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*70+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*74 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c86Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*71+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*75 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c86Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*72+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*76 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c87Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*73+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*77 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c87Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*74+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*78 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c88Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*75+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*79 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c88Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*76+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*80 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }}
  if(lMax > 8) { LNsNs=9*NsNs; LNs=9*Ns;
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
    sumMe = 0;/*c90*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*77+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*81 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c91Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*78+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*82 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c91Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*79+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*83 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c92Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*80+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*84 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c92Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*81+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*85 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c93Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*82+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*86 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c93Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*83+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*87 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c94Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*84+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*88 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c94Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*85+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*89 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c95Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*86+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*90 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c95Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*87+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*91 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c96Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*88+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*92 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c96Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*89+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*93 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c97Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*90+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*94 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c97Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*91+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*95 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c98Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*92+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*96 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c98Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*93+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*97 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c99Re*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*94+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*98 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
    sumMe = 0;/*c99Im*/ for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*95+i]);}
    for(int n = 0; n < Ns; n++){C[NsTsI + NsJ + Ns*99 + n] += bOa[LNsNs + n*Ns + k]*sumMe;}
  }}
////  double shiftBuffer = 96;
//  if(lMax > 9) {
//	 
//  LNsNs=10*NsNs; LNs=10*Ns; // OBS!!!!!! lMax > 9 Case!
//  for(int k = 0; k < Ns; k++){
//    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
//      for(int sumems = 100; sumems < 121; sumems++){
//        sumMe = 0; for(int i = 0; i < Asize; i++){
//		sumMe += exes[i]*(preCoef[totalAN*(sumems - 4)+i]);
////		printf("%d\n",totalAN*(sumems - 4)+i );
////		printf("total: %d\n",totalAN);
////		printf("preCoef: %f\n",preCoef[totalAN*(sumems - 4)+i]);
//	}
//        for(int n = 0; n < Ns; n++){
//		C[NsTsI + NsJ + Ns*sumems + n] += bOa[LNsNs + n*Ns + k]*sumMe; // FOUND SEGFAULT!!!!! OBS!!!
////		printf("preCoef: %f\n",bOa[LNsNs + n*Ns + k]); // FOUND HERE!!!
//	}
//	//shiftBuffer++; WRONG LOGIC, but not in use anyway ( not considering k++)
//      }
//  }}

  if(lMax > 9) {
  for(int restOfLs = 10; restOfLs <= lMax; restOfLs++){	 
  LNsNs=restOfLs*NsNs; LNs=restOfLs*Ns; // OBS!!!!!! lMax > 9 Case!
  for(int k = 0; k < Ns; k++){
    for(int i = 0; i < Asize; i++){exes[i] = exp(aOa[LNs + k]*r2[i]);}//exponents
      for(int sumems = restOfLs*restOfLs; sumems < (restOfLs+1)*(restOfLs+1); sumems++){
        sumMe = 0; for(int i = 0; i < Asize; i++){sumMe += exes[i]*(preCoef[totalAN*(sumems - 4)+i]);}
        for(int n = 0; n < Ns; n++){
		C[NsTsI + NsJ + Ns*sumems + n] += bOa[LNsNs + n*Ns + k]*sumMe; // FOUND SEGFAULT!!!!! OBS!!!
	}
      }
   }
  }}
}
//=======================================================================
/**
 * Used to calculate the partial power spectrum without crossover.
 */
void getPNoCrossD(double* soapMat, double* Cnnd, int Ns, int Ts, int Hs, int lMax){
  int NsTs100 = Ns*Ts*((lMax+1)*(lMax+1)); // Used to be NsTs100 = Ns*Ts*100, but 100 is a waste of memory if not lMax = 9, and can't do over that, so changed.
  int Ns100 = Ns*((lMax+1)*(lMax+1));
  int NsNs = (Ns*(Ns+1))/2;
  int NsNsLmax = NsNs*(lMax+1);
  int NsNsLmaxTs = NsNsLmax*Ts;
  int shiftN = 0;

  // The power spectrum is multiplied by an l-dependent prefactor that comes
  // from the normalization of the Wigner D matrices. This prefactor is
  // mentioned in the arrata of the original SOAP paper: On representing
  // chemical environments, Phys. Rev. B 87, 184115 (2013). Here the square
  // root of the prefactor in the dot-product kernel is used, so that after a
  // possible dot-product the full prefactor is recovered.

  double cs0=2.4674011003; double cs1=7.4022033011; double cs2=7.4022033005;
  // SUM M's UP!
  double prel0 = PI*sqrt(8.0/(1.0));
  for(int i = 0; i < Hs; i++){
    for(int j = 0; j < Ts; j++){
      shiftN = 0;
      for(int k = 0; k < Ns; k++){
        for(int kd = k; kd < Ns; kd++){
          soapMat[NsNsLmaxTs*i+ NsNsLmax*j+ 0 +shiftN] = prel0*(
            cs0*Cnnd[NsTs100*i + Ns100*j + 0 + k]*Cnnd[NsTs100*i + Ns100*j + 0*Ns + kd]);
          shiftN++;
        }
      }
    }
  } if(lMax > 0) {
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
  } if(lMax > 1) {
    double prel2 = PI*sqrt(8.0/(2.0*2.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 2*NsNs + shiftN] = prel2*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 4*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 4*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 5*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 5*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 6*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 6*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 7*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 7*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 8*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 8*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 2) {
    double prel3 = PI*sqrt(8.0/(2.0*3.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 3*NsNs + shiftN] = prel3*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 9*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 9*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 10*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 10*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 11*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 11*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 12*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 12*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 13*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 13*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 14*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 14*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 15*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 15*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 3) {
    double prel4 = PI*sqrt(8.0/(2.0*4.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 4*NsNs + shiftN] = prel4*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 16*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 16*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 17*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 17*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 18*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 18*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 19*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 19*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 20*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 20*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 21*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 21*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 22*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 22*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 23*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 23*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 24*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 24*Ns + kd]);
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
              PI3*Cnnd[NsTs100*i + Ns100*j + 25*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 25*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 26*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 26*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 27*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 27*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 28*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 28*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 29*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 29*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 30*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 30*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 31*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 31*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 32*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 32*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 33*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 33*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 34*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 34*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 35*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 35*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 5) {
    double prel6 = PI*sqrt(8.0/(2.0*6.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 6*NsNs + shiftN] = prel6*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 36*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 36*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 37*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 37*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 38*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 38*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 39*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 39*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 40*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 40*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 41*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 41*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 42*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 42*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 43*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 43*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 44*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 44*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 45*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 45*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 46*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 46*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 47*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 47*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 48*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 48*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 6) {
    double prel7 = PI*sqrt(8.0/(2.0*7.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 7*NsNs + shiftN] = prel7*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 49*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 49*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 50*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 50*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 51*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 51*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 52*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 52*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 53*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 53*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 54*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 54*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 55*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 55*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 56*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 56*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 57*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 57*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 58*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 58*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 59*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 59*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 60*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 60*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 61*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 61*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 62*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 62*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 63*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 63*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 7) {
    double prel8 = PI*sqrt(8.0/(2.0*8.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 8*NsNs + shiftN] = prel8*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 64*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 64*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 65*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 65*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 66*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 66*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 67*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 67*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 68*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 68*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 69*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 69*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 70*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 70*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 71*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 71*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 72*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 72*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 73*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 73*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 74*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 74*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 75*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 75*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 76*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 76*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 77*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 77*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 78*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 78*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 79*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 79*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 80*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 80*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 8) {
    double prel9 = PI*sqrt(8.0/(2.0*9.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 9*NsNs + shiftN] = prel9*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 81*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 81*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 82*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 82*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 83*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 83*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 84*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 84*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 85*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 85*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 86*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 86*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 87*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 87*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 88*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 88*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 89*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 89*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 90*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 90*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 91*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 91*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 92*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 92*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 93*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 93*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 94*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 94*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 95*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 95*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 96*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 96*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 97*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 97*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 98*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 98*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 99*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 99*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  }
   if(lMax > 9) { // OBS!!!! LMAX > 9 ------
    double prel10 = PI*sqrt(8.0/(2.0*10.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 100; buffShift < 121; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 10*NsNs + shiftN] = prel10*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 
   if(lMax > 10) { // OBS!!!! LMAX > 9 ------
    double prel11 = PI*sqrt(8.0/(2.0*11.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 11*11; buffShift < 12*12; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 11*NsNs + shiftN] = prel11*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 11) { // OBS!!!! LMAX > 9 ------
    double prel12 = PI*sqrt(8.0/(2.0*12.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 12*12; buffShift < 13*13; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 12*NsNs + shiftN] = prel12*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 12) { // OBS!!!! LMAX > 9 ------
    double prel13 = PI*sqrt(8.0/(2.0*13.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 13*13; buffShift < 14*14; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 13*NsNs + shiftN] = prel13*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 13) { // OBS!!!! LMAX > 9 ------
    double prel14 = PI*sqrt(8.0/(2.0*14.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 14*14; buffShift < 15*15; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 14*NsNs + shiftN] = prel14*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 14) { // OBS!!!! LMAX > 9 ------
    double prel15 = PI*sqrt(8.0/(2.0*15.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 15*15; buffShift < 16*16; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 15*NsNs + shiftN] = prel15*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 15) { // OBS!!!! LMAX > 9 ------
    double prel16 = PI*sqrt(8.0/(2.0*16.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 16*16; buffShift < 17*17; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 16*NsNs + shiftN] = prel16*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 16) { // OBS!!!! LMAX > 9 ------
    double prel17 = PI*sqrt(8.0/(2.0*17.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 17*17; buffShift < 18*18; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 17*NsNs + shiftN] = prel17*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 17) { // OBS!!!! LMAX > 9 ------
    double prel18 = PI*sqrt(8.0/(2.0*18.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 18*18; buffShift < 19*19; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 18*NsNs + shiftN] = prel18*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 18) { // OBS!!!! LMAX > 9 ------
    double prel19 = PI*sqrt(8.0/(2.0*19.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 19*19; buffShift < 20*20; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMat[NsNsLmaxTs*i+NsNsLmax*j+ 19*NsNs + shiftN] = prel19*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 
}
//=======================================================================
//=======================================================================
/**
 * Used to calculate the partial power spectrum derivative without crossover.
 */
  void getPNoCrossDevX(double* soapMatDevX,double* Cdev, double* Cnnd,  int Ns, int Ts, int Hs, int lMax, int totalAN){
  int NsTs100 = Ns*Ts*((lMax+1)*(lMax+1)); // Used to be NsTs100 = Ns*Ts*100, but 100 is a waste of memory if not lMax = 9, and can't do over that, so changed.
  int Ns100 = Ns*((lMax+1)*(lMax+1));
  int NsNs = (Ns*(Ns+1))/2;
  int NsNsLmax = NsNs*(lMax+1);
  int NsNsLmaxTs = NsNsLmax*Ts;
  int shiftN = 0;

  // The power spectrum is multiplied by an l-dependent prefactor that comes
  // from the normalization of the Wigner D matrices. This prefactor is
  // mentioned in the arrata of the original SOAP paper: On representing
  // chemical environments, Phys. Rev. B 87, 184115 (2013). Here the square
  // root of the prefactor in the dot-product kernel is used, so that after a
  // possible dot-product the full prefactor is recovered.
  std::cout << "START!" << std::endl;

  double cs0=2.4674011003; double cs1=7.4022033011; double cs2=7.4022033005;
  // SUM M's UP!
  double prel0 = PI*sqrt(8.0/(1.0));
  for(int a = 0; a < totalAN; a++){
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMatDevX[NsNsLmaxTs*i*totalAN+ NsNsLmax*j*totalAN+ 0*totalAN +shiftN*totalAN + a] = prel0*(cs0*Cnnd[NsTs100*i + Ns100*j + 0 + k]*Cdev[NsTs100*i*totalAN + Ns100*j*totalAN + 0*Ns*totalAN + kd*totalAN + a]);
	    std::cout << soapMatDevX[NsNsLmaxTs*i*totalAN+ NsNsLmax*j*totalAN+ 0*totalAN +shiftN*totalAN + a]  << std::endl; 
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 0) {
    double prel1 = PI*sqrt(8.0/(2.0*1.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ NsNs + shiftN] = prel1*(
              cs1*Cnnd[NsTs100*i + Ns100*j + 1*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 1*Ns + kd]
             +cs2*Cnnd[NsTs100*i + Ns100*j + 2*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 2*Ns + kd]
             +cs2*Cnnd[NsTs100*i + Ns100*j + 3*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 3*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 1) {
    double prel2 = PI*sqrt(8.0/(2.0*2.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 2*NsNs + shiftN] = prel2*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 4*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 4*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 5*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 5*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 6*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 6*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 7*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 7*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 8*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 8*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 2) {
    double prel3 = PI*sqrt(8.0/(2.0*3.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 3*NsNs + shiftN] = prel3*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 9*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 9*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 10*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 10*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 11*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 11*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 12*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 12*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 13*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 13*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 14*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 14*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 15*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 15*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 3) {
    double prel4 = PI*sqrt(8.0/(2.0*4.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 4*NsNs + shiftN] = prel4*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 16*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 16*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 17*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 17*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 18*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 18*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 19*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 19*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 20*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 20*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 21*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 21*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 22*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 22*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 23*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 23*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 24*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 24*Ns + kd]);
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
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 5*NsNs + shiftN] = prel5*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 25*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 25*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 26*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 26*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 27*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 27*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 28*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 28*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 29*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 29*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 30*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 30*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 31*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 31*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 32*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 32*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 33*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 33*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 34*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 34*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 35*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 35*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 5) {
    double prel6 = PI*sqrt(8.0/(2.0*6.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 6*NsNs + shiftN] = prel6*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 36*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 36*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 37*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 37*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 38*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 38*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 39*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 39*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 40*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 40*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 41*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 41*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 42*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 42*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 43*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 43*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 44*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 44*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 45*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 45*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 46*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 46*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 47*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 47*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 48*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 48*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 6) {
    double prel7 = PI*sqrt(8.0/(2.0*7.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 7*NsNs + shiftN] = prel7*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 49*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 49*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 50*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 50*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 51*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 51*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 52*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 52*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 53*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 53*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 54*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 54*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 55*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 55*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 56*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 56*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 57*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 57*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 58*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 58*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 59*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 59*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 60*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 60*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 61*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 61*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 62*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 62*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 63*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 63*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 7) {
    double prel8 = PI*sqrt(8.0/(2.0*8.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 8*NsNs + shiftN] = prel8*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 64*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 64*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 65*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 65*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 66*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 66*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 67*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 67*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 68*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 68*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 69*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 69*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 70*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 70*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 71*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 71*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 72*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 72*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 73*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 73*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 74*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 74*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 75*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 75*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 76*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 76*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 77*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 77*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 78*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 78*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 79*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 79*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 80*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 80*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  } if(lMax > 8) {
    double prel9 = PI*sqrt(8.0/(2.0*9.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 9*NsNs + shiftN] = prel9*(
              PI3*Cnnd[NsTs100*i + Ns100*j + 81*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 81*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 82*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 82*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 83*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 83*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 84*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 84*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 85*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 85*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 86*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 86*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 87*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 87*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 88*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 88*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 89*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 89*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 90*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 90*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 91*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 91*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 92*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 92*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 93*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 93*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 94*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 94*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 95*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 95*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 96*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 96*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 97*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 97*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 98*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 98*Ns + kd]
             +PI3*Cnnd[NsTs100*i + Ns100*j + 99*Ns + k]*Cnnd[NsTs100*i + Ns100*j + 99*Ns + kd]);
            shiftN++;
          }
        }
      }
    }
  }
   if(lMax > 9) { // OBS!!!! LMAX > 9 ------
    double prel10 = PI*sqrt(8.0/(2.0*10.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 100; buffShift < 121; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 10*NsNs + shiftN] = prel10*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 
   if(lMax > 10) { // OBS!!!! LMAX > 9 ------
    double prel11 = PI*sqrt(8.0/(2.0*11.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 11*11; buffShift < 12*12; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 11*NsNs + shiftN] = prel11*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 11) { // OBS!!!! LMAX > 9 ------
    double prel12 = PI*sqrt(8.0/(2.0*12.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 12*12; buffShift < 13*13; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 12*NsNs + shiftN] = prel12*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 12) { // OBS!!!! LMAX > 9 ------
    double prel13 = PI*sqrt(8.0/(2.0*13.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 13*13; buffShift < 14*14; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 13*NsNs + shiftN] = prel13*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 13) { // OBS!!!! LMAX > 9 ------
    double prel14 = PI*sqrt(8.0/(2.0*14.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 14*14; buffShift < 15*15; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 14*NsNs + shiftN] = prel14*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 14) { // OBS!!!! LMAX > 9 ------
    double prel15 = PI*sqrt(8.0/(2.0*15.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 15*15; buffShift < 16*16; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 15*NsNs + shiftN] = prel15*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 15) { // OBS!!!! LMAX > 9 ------
    double prel16 = PI*sqrt(8.0/(2.0*16.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 16*16; buffShift < 17*17; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 16*NsNs + shiftN] = prel16*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 16) { // OBS!!!! LMAX > 9 ------
    double prel17 = PI*sqrt(8.0/(2.0*17.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 17*17; buffShift < 18*18; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 17*NsNs + shiftN] = prel17*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 17) { // OBS!!!! LMAX > 9 ------
    double prel18 = PI*sqrt(8.0/(2.0*18.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 18*18; buffShift < 19*19; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 18*NsNs + shiftN] = prel18*buffDouble;
            shiftN++;
          }
        }
      }
    }
  } 

   if(lMax > 18) { // OBS!!!! LMAX > 9 ------
    double prel19 = PI*sqrt(8.0/(2.0*19.0+1.0));
    for(int i = 0; i < Hs; i++){
      for(int j = 0; j < Ts; j++){
        shiftN = 0;
        for(int k = 0; k < Ns; k++){
          for(int kd = k; kd < Ns; kd++){
            double buffDouble = 0;
            for(int buffShift = 19*19; buffShift < 20*20; buffShift++){
              buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
	    }
            soapMatDevX[NsNsLmaxTs*i+NsNsLmax*j+ 19*NsNs + shiftN] = prel19*buffDouble;
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
void getPCrossOverD(double* soapMat, double* Cnnd, int Ns, int Ts, int Hs, int lMax){
  int NsTs100 = Ns*Ts*((lMax+1)*(lMax+1));
  int Ns100 = Ns*((lMax+1)*(lMax+1));
  int NsNs = (Ns*(Ns+1))/2;
  int NsNsLmax = NsNs*(lMax+1) ;
  int NsNsLmaxTs = NsNsLmax*getCrosNumD(Ts);
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
//  double cs3=3.0842513755; double cs4=37.0110165048; double cs5=9.2527541262;
//  double cs6=4.3179519254; double cs7=6.4769278880; double cs8=64.7692788826;
//  double cs9=10.7948798139; double cs10=0.3469782797; double cs11=13.8791311886;
//  double cs12=6.9395655942; double cs13=97.1539183221; double cs14=12.1442397908;
//  double cs15=0.4240845642; double cs16=6.3612684614; double cs17=178.1155169268;
//  double cs18=7.4214798715; double cs19=133.5866376943; double cs20=13.3586637700;
//  double cs21=0.1252977121; double cs22=10.5250078181; double cs23=6.5781298867;
//  double cs24=26.3125195455; double cs25=7.8937558638; double cs26=173.6626290026;
//  double cs27=14.4718857505; double cs28=0.1445742832; double cs29=0.2530049956;
//  double cs30=1.5180299739; double cs31=0.7590149869; double cs32=33.3966594236;
//  double cs33=8.3491648559; double cs34=217.0782862628; double cs35=15.5055918758;
//  double cs36=0.0025601696; double cs37=0.3686644222; double cs38=6.4516273890;
//  double cs39=47.3119341841; double cs40=7.0967901272; double cs41=369.0330866085;
//  double cs42=8.7865020627; double cs43=263.5950618888; double cs44=16.4746913675;
//  double cs45=0.0028613660; double cs46=0.1287614710; double cs47=11.3310094474;
//  double cs48=6.6097555108; double cs49=515.5609298206; double cs50=7.3651561406;
//  double cs51=49.1010409380; double cs52=9.2064451758; double cs53=313.0191359728;
//  double cs54=17.3899519988;

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
                  PI3*Cnnd[NsTs100*i + Ns100*j + 4*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 4*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 5*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 5*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 6*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 6*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 7*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 7*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 8*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 8*Ns + kd]);
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
                  PI3*Cnnd[NsTs100*i + Ns100*j + 9*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 9*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 10*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 10*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 11*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 11*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 12*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 12*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 13*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 13*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 14*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 14*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 15*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 15*Ns + kd]);
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
                  PI3*Cnnd[NsTs100*i + Ns100*j + 16*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 16*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 17*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 17*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 18*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 18*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 19*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 19*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 20*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 20*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 21*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 21*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 22*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 22*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 23*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 23*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 24*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 24*Ns + kd]);
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
                  PI3*Cnnd[NsTs100*i + Ns100*j + 25*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 25*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 26*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 26*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 27*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 27*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 28*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 28*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 29*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 29*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 30*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 30*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 31*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 31*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 32*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 32*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 33*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 33*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 34*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 34*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 35*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 35*Ns + kd]);
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
                  PI3*Cnnd[NsTs100*i + Ns100*j + 36*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 36*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 37*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 37*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 38*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 38*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 39*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 39*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 40*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 40*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 41*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 41*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 42*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 42*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 43*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 43*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 44*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 44*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 45*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 45*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 46*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 46*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 47*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 47*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 48*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 48*Ns + kd]);
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
                  PI3*Cnnd[NsTs100*i + Ns100*j + 49*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 49*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 50*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 50*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 51*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 51*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 52*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 52*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 53*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 53*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 54*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 54*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 55*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 55*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 56*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 56*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 57*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 57*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 58*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 58*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 59*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 59*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 60*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 60*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 61*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 61*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 62*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 62*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 63*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 63*Ns + kd]);
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
                  PI3*Cnnd[NsTs100*i + Ns100*j + 64*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 64*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 65*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 65*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 66*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 66*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 67*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 67*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 68*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 68*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 69*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 69*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 70*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 70*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 71*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 71*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 72*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 72*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 73*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 73*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 74*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 74*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 75*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 75*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 76*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 76*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 77*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 77*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 78*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 78*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 79*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 79*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 80*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 80*Ns + kd]);
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
                  PI3*Cnnd[NsTs100*i + Ns100*j + 81*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 81*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 82*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 82*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 83*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 83*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 84*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 84*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 85*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 85*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 86*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 86*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 87*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 87*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 88*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 88*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 89*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 89*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 90*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 90*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 91*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 91*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 92*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 92*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 93*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 93*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 94*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 94*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 95*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 95*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 96*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 96*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 97*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 97*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 98*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 98*Ns + kd]
                  +PI3*Cnnd[NsTs100*i + Ns100*j + 99*Ns + k]*Cnnd[NsTs100*i + Ns100*jd + 99*Ns + kd]);
              shiftN++;
            }
          }
          shiftT++;
        }
      }
    }
  }

   if (lMax > 9) { // OBS!!!! LMAX > 9 ------
    double prel10 = PI*sqrt(8.0/(2.0*10.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = 100; buffShift < 121; buffShift++){
                buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
  	       }
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 10*NsNs + shiftN] = prel10*buffDouble;
              shiftN++;
            }
          }
	  shiftT++;
        }
      }
    }
  } 

   if (lMax > 10) { // OBS!!!! LMAX > 9 ------
    double prel11 = PI*sqrt(8.0/(2.0*11.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = 11*11; buffShift < 12*12; buffShift++){
                buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
  	       }
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 11*NsNs + shiftN] = prel11*buffDouble;
              shiftN++;
            }
          }
	  shiftT++;
        }
      }
    }
  } 

   if (lMax > 11) { // OBS!!!! LMAX > 9 ------
    double prel12 = PI*sqrt(8.0/(2.0*12.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = 12*12; buffShift < 13*13; buffShift++){
                buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
  	       }
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 12*NsNs + shiftN] = prel12*buffDouble;
              shiftN++;
            }
          }
	  shiftT++;
        }
      }
    }
  } 

   if (lMax > 12) { // OBS!!!! LMAX > 9 ------
    double prel13 = PI*sqrt(8.0/(2.0*13.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = 13*13; buffShift < 14*14; buffShift++){
                buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
  	       }
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 13*NsNs + shiftN] = prel13*buffDouble;
              shiftN++;
            }
          }
	  shiftT++;
        }
      }
    }
  } 

   if (lMax > 13) { // OBS!!!! LMAX > 9 ------
    double prel14 = PI*sqrt(8.0/(2.0*14.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = 14*14; buffShift < 15*15; buffShift++){
                buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
  	       }
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 14*NsNs + shiftN] = prel14*buffDouble;
              shiftN++;
            }
          }
	  shiftT++;
        }
      }
    }
  } 

   if (lMax > 14) { // OBS!!!! LMAX > 9 ------
    double prel15 = PI*sqrt(8.0/(2.0*15.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = 15*15; buffShift < 16*16; buffShift++){
                buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
  	       }
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 15*NsNs + shiftN] = prel15*buffDouble;
              shiftN++;
            }
          }
	  shiftT++;
        }
      }
    }
  } 

   if (lMax > 15) { // OBS!!!! LMAX > 9 ------
    double prel16 = PI*sqrt(8.0/(2.0*16.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = 16*16; buffShift < 17*17; buffShift++){
                buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
  	       }
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 16*NsNs + shiftN] = prel16*buffDouble;
              shiftN++;
            }
          }
	  shiftT++;
        }
      }
    }
  } 

   if (lMax > 16) { // OBS!!!! LMAX > 9 ------
    double prel17 = PI*sqrt(8.0/(2.0*17.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = 17*17; buffShift < 18*18; buffShift++){
                buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
  	       }
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 17*NsNs + shiftN] = prel17*buffDouble;
              shiftN++;
            }
          }
	  shiftT++;
        }
      }
    }
  } 

   if (lMax > 17) { // OBS!!!! LMAX > 9 ------
    double prel18 = PI*sqrt(8.0/(2.0*18.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = 18*18; buffShift < 19*19; buffShift++){
                buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
  	       }
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 18*NsNs + shiftN] = prel18*buffDouble;
              shiftN++;
            }
          }
	  shiftT++;
        }
      }
    }
  } 

   if (lMax > 18) { // OBS!!!! LMAX > 9 ------
    double prel19 = PI*sqrt(8.0/(2.0*19.0+1.0));
    for(int i = 0; i < Hs; i++){
      shiftT = 0;
      for(int j = 0; j < Ts; j++){
        for(int jd = j; jd < Ts; jd++){
          shiftN = 0;
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = 19*19; buffShift < 20*20; buffShift++){
                buffDouble += PI3*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + k]*Cnnd[NsTs100*i + Ns100*j + buffShift*Ns + kd];
  	       }
              soapMat[NsNsLmaxTs*i+NsNsLmax*shiftT+ 19*NsNs + shiftN] = prel19*buffDouble;
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
void soapGTODevX(py::array_t<double> cArr, py::array_t<double> positions, py::array_t<double> HposArr, py::array_t<double> alphasArr, py::array_t<double> betasArr, py::array_t<int> atomicNumbersArr, double rCut, double cutoffPadding, int totalAN, int Nt, int Ns, int lMax, int Hs, double eta, bool crossover) {

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
  double* x2 = (double*) malloc(sizeof(double)*totalAN);
  double* x4 = (double*) malloc(sizeof(double)*totalAN);
  double* x6 = (double*) malloc(sizeof(double)*totalAN);
  double* x8 = (double*) malloc(sizeof(double)*totalAN);
  double* x10 = (double*) malloc(sizeof(double)*totalAN);
  double* x12 = (double*) malloc(sizeof(double)*totalAN);
  double* x14 = (double*) malloc(sizeof(double)*totalAN);
  double* x16 = (double*) malloc(sizeof(double)*totalAN);
  double* x18 = (double*) malloc(sizeof(double)*totalAN);
  double* y2 = (double*) malloc(sizeof(double)*totalAN);
  double* y4 = (double*) malloc(sizeof(double)*totalAN);
  double* y6 = (double*) malloc(sizeof(double)*totalAN);
  double* y8 = (double*) malloc(sizeof(double)*totalAN);
  double* y10 = (double*) malloc(sizeof(double)*totalAN);
  double* y12 = (double*) malloc(sizeof(double)*totalAN);
  double* y14 = (double*) malloc(sizeof(double)*totalAN);
  double* y16 = (double*) malloc(sizeof(double)*totalAN);
  double* y18 = (double*) malloc(sizeof(double)*totalAN);
  double* z2 = (double*) malloc(sizeof(double)*totalAN);
  double* z4 = (double*) malloc(sizeof(double)*totalAN);
  double* z6 = (double*) malloc(sizeof(double)*totalAN);
  double* z8 = (double*) malloc(sizeof(double)*totalAN);
  double* z10 = (double*) malloc(sizeof(double)*totalAN);
  double* z12 = (double*) malloc(sizeof(double)*totalAN);
  double* z14 = (double*) malloc(sizeof(double)*totalAN);
  double* z16 = (double*) malloc(sizeof(double)*totalAN);
  double* z18 = (double*) malloc(sizeof(double)*totalAN);
  double* r2 = (double*) malloc(sizeof(double)*totalAN);
  double* r4 = (double*) malloc(sizeof(double)*totalAN);
  double* r6 = (double*) malloc(sizeof(double)*totalAN);
  double* r8 = (double*) malloc(sizeof(double)*totalAN);
  double* r10 = (double*) malloc(sizeof(double)*totalAN);
  double* r12 = (double*) malloc(sizeof(double)*totalAN);
  double* r14 = (double*) malloc(sizeof(double)*totalAN);
  double* r16 = (double*) malloc(sizeof(double)*totalAN);
  double* r18 = (double*) malloc(sizeof(double)*totalAN);
  double* exes = (double*) malloc (sizeof(double)*totalAN);
  double* preCoef = (double*) malloc(((lMax + 1)*(lMax + 1) - 4)*sizeof(double)*totalAN); // -4 because no need for l=0 and l=1 (m=-1,0,1) cases.
  double* bOa = (double*) malloc((lMax+1)*NsNs*sizeof(double));
  double* aOa = (double*) malloc((lMax+1)*Ns*sizeof(double));


  double* cnnd = (double*) malloc(((lMax+1)*(lMax+1))*Nt*Ns*Hs*sizeof(double));
  double* cdevX = (double*) malloc(totalAN*((lMax+1)*(lMax+1))*Nt*Ns*Hs*sizeof(double));
  for(int i = 0; i < ((lMax+1)*(lMax+1))*Nt*Ns*Hs; i++){cnnd[i] = 0.0;}
  for(int i = 0; i < ((lMax+1)*(lMax+1))*Nt*Ns*Hs*totalAN; i++){cdevX[i] = 0.0;}

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

  getAlphaBetaD(aOa,bOa,alphas,betas,Ns,lMax,oOeta, oOeta3O2);

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
      getDeltaD(dx, dy, dz, positions, ix, iy, iz, ZIndexPair.second);

      getRsZsD(dx,x2,x4,x6,x8,x10,x12,x14,x16,x18, dy,y2,y4,y6,y8,y10,y12,y14,y16,y18, dz, r2, r4, r6, r8,r10,r12,r14,r16,r18, z2, z4, z6, z8,z10,z12,z14,z16,z18, n_neighbours,lMax);
      getCfactorsD(preCoef, n_neighbours, dx,x2, x4, x6, x8,x10,x12,x14,x16,x18, dy,y2, y4, y6, y8,y10,y12,y14,y16,y18, dz, z2, z4, z6, z8,z10,z12,z14,z16,z18, r2, r4, r6, r8,r10,r12,r14,r16,r18, totalAN, lMax); // Erased tn
      getCD(cdevX,cnnd, preCoef, dx, dy, dz, r2, bOa, aOa, exes, totalAN, n_neighbours, Ns, Nt, lMax, i, j, ZIndexPair.second); //erased tn and Nx
    }
  }

  free(dx);
  free(x2);
  free(x4);
  free(x6);
  free(x8);
  free(x10);
  free(x12);
  free(x14);
  free(x16);
  free(x18);
  free(dy);
  free(y2);
  free(y4);
  free(y6);
  free(y8);
  free(y10);
  free(y12);
  free(y14);
  free(y16);
  free(y18);
  free(dz);
  free(z2);
  free(z4);
  free(z6);
  free(z8);
  free(z10);
  free(z12);
  free(z14);
  free(z16);
  free(z18);
  free(r2);
  free(r4);
  free(r6);
  free(r8);
  free(r10);
  free(r12);
  free(r14);
  free(r16);
  free(r18);
  free(exes);
  free(preCoef);
  free(bOa);
  free(aOa);

  if (crossover) {
//    getPCrossOverD(c, cnnd, Ns, Nt, Hs, lMax);
    getPNoCrossDevX(c,cdevX, cnnd, Ns, Nt, Hs, lMax, totalAN);
  } else {
    getPNoCrossDevX(c,cdevX, cnnd, Ns, Nt, Hs, lMax, totalAN);
  };
  free(cnnd);
 
  return;
}

