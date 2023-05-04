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
#include "soapGTO.h"
#include "celllist.h"
#include "weighting.h"

#define PI2 9.86960440108936
#define PI 3.141592653589793238
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
inline void getRsZsD(double* x, double* x2, double* x4, double* x6, double* x8, double* x10, double* x12, double* x14, double* x16, double* x18, double* y, double* y2, double* y4, double* y6, double* y8, double* y10, double* y12, double* y14, double* y16, double* y18, double* z, double* r2, double* r4, double* r6, double* r8, double* r10, double* r12, double* r14, double* r16, double* r18, double* z2, double* z4, double* z6, double* z8, double* z10, double* z12, double* z14, double* z16, double* z18, double* r20, double* x20, double* y20, double* z20, int size, int lMax) {
  double xx;
  double yy;
  double zz;
  double rr;

  for (int i = 0; i < size; i++) {
    xx = x[i]*x[i];
    yy = y[i]*y[i];
    zz = z[i]*z[i];
    rr = xx + yy + zz;
    x2[i] = xx;
    y2[i] = yy;
    z2[i] = zz;
    r2[i] = rr;

    if(lMax > 3){ r4[i] = r2[i]*r2[i]; z4[i] = z2[i]*z2[i]; x4[i] = x2[i]*x2[i]; y4[i] = y2[i]*y2[i];
      if(lMax > 5){ r6[i] = r2[i]*r4[i]; z6[i] = z2[i]*z4[i]; x6[i] = x2[i]*x4[i]; y6[i] = y2[i]*y4[i];
        if(lMax > 7){ r8[i] = r4[i]*r4[i]; z8[i] = z4[i]*z4[i]; x8[i] = x4[i]*x4[i]; y8[i] = y4[i]*y4[i];
          if(lMax > 9){ x10[i] = x6[i]*x4[i]; y10[i] = y6[i]*y4[i]; z10[i] = z6[i]*z4[i]; r10[i] = r6[i]*r4[i];
            if(lMax > 11){ x12[i] = x6[i]*x6[i]; y12[i] = y6[i]*y6[i]; r12[i] = r6[i]*r6[i]; z12[i] = z6[i]*z6[i];
              if(lMax > 13){ x14[i] = x6[i]*x8[i]; y14[i] = y6[i]*y8[i]; r14[i] = r6[i]*r8[i]; z14[i] = z6[i]*z8[i];
                if(lMax > 15){ x16[i] = x8[i]*x8[i]; y16[i] = y8[i]*y8[i]; r16[i] = r8[i]*r8[i]; z16[i] = z8[i]*z8[i];
                  if(lMax > 17){ x18[i] = x10[i]*x8[i]; y18[i] = y10[i]*y8[i]; r18[i] = r10[i]*r8[i]; z18[i] = z10[i]*z8[i];
                   if(lMax > 19){ x20[i] = x10[i]*x10[i];  z20[i] = z10[i]*z10[i];y20[i] = y10[i]*y10[i];r20[i] = r10[i]*r10[i];
                  }
                }
              }
            }
          }
	    }
      }
    }
  }
}}
//================================================================
void getAlphaBetaD(double* aOa, double* bOa, double* alphas, double* betas, int Ns, int lMax, double oOeta, double oOeta3O2) {

  int NsNs = Ns*Ns;
  double oneO1alpha;
  double oneO1alpha2;
  double oneO1alphaSqrt;
  double oneO1alphaSqrtX;

  for (int myL = 0; myL < lMax + 1 ;myL++) {
    for (int k = 0; k < Ns; k++) {
      oneO1alpha = 1.0/(1.0 + oOeta*alphas[myL*Ns + k]);
      oneO1alphaSqrt = sqrt(oneO1alpha);
      aOa[myL*Ns + k] = -alphas[myL*Ns + k]*oneO1alpha; 
      oneO1alpha2 = pow(oneO1alpha, myL+1);
      oneO1alphaSqrtX = oneO1alphaSqrt*oneO1alpha2;
      for (int n = 0; n < Ns; n++) {
        bOa[myL*NsNs + n*Ns + k] = oOeta3O2*betas[myL*NsNs + n*Ns + k]*oneO1alphaSqrtX;
      } 
    }
  }
}
//================================================================
void getCfactorsD(double* preCoef, double* prCofDX, double* prCofDY, double* prCofDZ, int Asize, double* x,double* x2, double* x4, double* x6, double* x8, double* x10,double* x12,double* x14,double* x16,double* x18, double* y,double* y2, double* y4, double* y6, double* y8, double* y10,double* y12,double* y14,double* y16,double* y18, double* z, double* z2, double* z4, double* z6, double* z8, double* z10,double* z12,double* z14,double* z16,double* z18, double* r2, double* r4, double* r6, double* r8,double* r10,double* r12,double* r14,double* r16,double* r18, double* r20,  double* x20,  double* y20,  double* z20, int totalAN, int lMax, bool return_derivatives){

  for (int i = 0; i < Asize; i++) {
    if (lMax > 1){
        preCoef[         +i] = 1.09254843059208*x[i]*y[i];
        preCoef[totalAN*1+i] = 1.09254843059208*y[i]*z[i];
        preCoef[totalAN*2+i] = -0.31539156525252*x2[i] - 0.31539156525252*y2[i] + 0.63078313050504*z2[i];
        preCoef[totalAN*3+i] = 1.09254843059208*x[i]*z[i];
        preCoef[totalAN*4+i] = 0.54627421529604*x2[i] - 0.54627421529604*y2[i];
        if (return_derivatives) {
        prCofDX[         +i] = 1.09254843059208*y[i];
        prCofDX[totalAN*1+i] = 0;
        prCofDX[totalAN*2+i] = -0.63078313050504*x[i];
        prCofDX[totalAN*3+i] = 1.09254843059208*z[i];
        prCofDX[totalAN*4+i] = 1.09254843059208*x[i];

        prCofDY[         +i] = 1.09254843059208*x[i];
        prCofDY[totalAN*1+i] = 1.09254843059208*z[i];
        prCofDY[totalAN*2+i] = -0.63078313050504*y[i];
        prCofDY[totalAN*3+i] = 0;
        prCofDY[totalAN*4+i] = -1.09254843059208*y[i];

        prCofDZ[         +i] = 0;
        prCofDZ[totalAN*1+i] = 1.09254843059208*y[i];
        prCofDZ[totalAN*2+i] = 1.26156626101008*z[i];
        prCofDZ[totalAN*3+i] = 1.09254843059208*x[i];
        prCofDZ[totalAN*4+i] = 0;
        }
    if (lMax > 2){
        preCoef[totalAN*5+i] = 0.590043589926644*y[i]*(3.0*x2[i] - y2[i]);
        preCoef[totalAN*6+i] = 2.89061144264055*x[i]*y[i]*z[i];
        preCoef[totalAN*7+i] = -0.457045799464466*y[i]*(x2[i] + y2[i] - 4.0*z2[i]);
        preCoef[totalAN*8+i] = 0.373176332590115*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 2.0*z2[i]);
        preCoef[totalAN*9+i] = -0.457045799464466*x[i]*(x2[i] + y2[i] - 4.0*z2[i]);
        preCoef[totalAN*10+i] = 1.44530572132028*z[i]*(x2[i] - y2[i]);
        preCoef[totalAN*11+i] = 0.590043589926644*x[i]*(x2[i] - 3.0*y2[i]);

        if(return_derivatives){
        prCofDX[totalAN*5+i] = 3.54026153955986*x[i]*y[i];
        prCofDX[totalAN*6+i] = 2.89061144264055*y[i]*z[i];
        prCofDX[totalAN*7+i] = -0.914091598928931*x[i]*y[i];
        prCofDX[totalAN*8+i] = -2.23905799554069*x[i]*z[i];
        prCofDX[totalAN*9+i] = -1.3711373983934*x2[i] - 0.457045799464466*y2[i] + 1.82818319785786*z2[i];
        prCofDX[totalAN*10+i] = 2.89061144264055*x[i]*z[i];
        prCofDX[totalAN*11+i] = 1.77013076977993*x2[i] - 1.77013076977993*y2[i];

        prCofDY[totalAN*5+i] = 1.77013076977993*x2[i] - 1.77013076977993*y2[i];
        prCofDY[totalAN*6+i] = 2.89061144264055*x[i]*z[i];
        prCofDY[totalAN*7+i] = -0.457045799464466*x2[i] - 1.3711373983934*y2[i] + 1.82818319785786*z2[i];
        prCofDY[totalAN*8+i] = -2.23905799554069*y[i]*z[i];
        prCofDY[totalAN*9+i] = -0.914091598928931*x[i]*y[i];
        prCofDY[totalAN*10+i] = -2.89061144264055*y[i]*z[i];
        prCofDY[totalAN*11+i] = -3.54026153955986*x[i]*y[i];

        prCofDZ[totalAN*5+i] = 0;
        prCofDZ[totalAN*6+i] = 2.89061144264055*x[i]*y[i];
        prCofDZ[totalAN*7+i] = 3.65636639571573*y[i]*z[i];
        prCofDZ[totalAN*8+i] = -1.11952899777035*x2[i] - 1.11952899777035*y2[i] + 2.23905799554069*z2[i];
        prCofDZ[totalAN*9+i] = 3.65636639571573*x[i]*z[i];
        prCofDZ[totalAN*10+i] = 1.44530572132028*x2[i] - 1.44530572132028*y2[i];
        prCofDZ[totalAN*11+i] = 0;
        }
    if (lMax > 3){
        preCoef[totalAN*12+i] = 2.5033429417967*x[i]*y[i]*(x2[i] - y2[i]);
        preCoef[totalAN*13+i] = 1.77013076977993*y[i]*z[i]*(3.0*x2[i] - y2[i]);
        preCoef[totalAN*14+i] = -0.94617469575756*x[i]*y[i]*(x2[i] + y2[i] - 6.0*z2[i]);
        preCoef[totalAN*15+i] = 0.669046543557289*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 4.0*z2[i]);
        preCoef[totalAN*16+i] = 3.70249414203215*z4[i] - 3.17356640745613*z2[i]*r2[i] + 0.317356640745613*r4[i];
        preCoef[totalAN*17+i] = 0.669046543557289*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 4.0*z2[i]);
        preCoef[totalAN*18+i] = -0.47308734787878*(x2[i] - y2[i])*(x2[i] + y2[i] - 6.0*z2[i]);
        preCoef[totalAN*19+i] = 1.77013076977993*x[i]*z[i]*(x2[i] - 3.0*y2[i]);
        preCoef[totalAN*20+i] = 0.625835735449176*x4[i] - 3.75501441269506*x2[i]*y2[i] + 0.625835735449176*y4[i];

        if(return_derivatives){
        prCofDX[totalAN*12+i] = 2.5033429417967*y[i]*(3.0*x2[i] - y2[i]);
        prCofDX[totalAN*13+i] = 10.6207846186796*x[i]*y[i]*z[i];
        prCofDX[totalAN*14+i] = 0.94617469575756*y[i]*(-3.0*x2[i] - y2[i] + 6.0*z2[i]);
        prCofDX[totalAN*15+i] = -4.01427926134374*x[i]*y[i]*z[i];
        prCofDX[totalAN*16+i] = 1.26942656298245*x[i]*(x2[i] + y2[i] - 4.0*z2[i]);
        prCofDX[totalAN*17+i] = 0.669046543557289*z[i]*(-9.0*x2[i] - 3.0*y2[i] + 4.0*z2[i]);
        prCofDX[totalAN*18+i] = 1.89234939151512*x[i]*(-x2[i] + 3.0*z2[i]);
        prCofDX[totalAN*19+i] = 5.31039230933979*z[i]*(x2[i] - y2[i]);
        prCofDX[totalAN*20+i] = 2.5033429417967*x[i]*(x2[i] - 3.0*y2[i]);

        prCofDY[totalAN*12+i] = 2.5033429417967*x[i]*(x2[i] - 3.0*y2[i]);
        prCofDY[totalAN*13+i] = 5.31039230933979*z[i]*(x2[i] - y2[i]);
        prCofDY[totalAN*14+i] = 0.94617469575756*x[i]*(-x2[i] - 3.0*y2[i] + 6.0*z2[i]);
        prCofDY[totalAN*15+i] = 0.669046543557289*z[i]*(-3.0*x2[i] - 9.0*y2[i] + 4.0*z2[i]);
        prCofDY[totalAN*16+i] = 1.26942656298245*y[i]*(x2[i] + y2[i] - 4.0*z2[i]);
        prCofDY[totalAN*17+i] = -4.01427926134374*x[i]*y[i]*z[i];
        prCofDY[totalAN*18+i] = 1.89234939151512*y[i]*(y2[i] - 3.0*z2[i]);
        prCofDY[totalAN*19+i] = -10.6207846186796*x[i]*y[i]*z[i];
        prCofDY[totalAN*20+i] = 2.5033429417967*y[i]*(-3.0*x2[i] + y2[i]);

        prCofDZ[totalAN*12+i] = 0;
        prCofDZ[totalAN*13+i] = 1.77013076977993*y[i]*(3.0*x2[i] - y2[i]);
        prCofDZ[totalAN*14+i] = 11.3540963490907*x[i]*y[i]*z[i];
        prCofDZ[totalAN*15+i] = -2.00713963067187*y[i]*(x2[i] + y2[i] - 4.0*z2[i]);
        prCofDZ[totalAN*16+i] = 1.69256875064327*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 2.0*z2[i]);
        prCofDZ[totalAN*17+i] = -2.00713963067187*x[i]*(x2[i] + y2[i] - 4.0*z2[i]);
        prCofDZ[totalAN*18+i] = 5.67704817454536*z[i]*(x2[i] - y2[i]);
        prCofDZ[totalAN*19+i] = 1.77013076977993*x[i]*(x2[i] - 3.0*y2[i]);
        prCofDZ[totalAN*20+i] = 0;
        }
    if (lMax > 4){
        preCoef[totalAN*21+i] = 0.65638205684017*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
        preCoef[totalAN*22+i] = 8.30264925952416*x[i]*y[i]*z[i]*(x2[i] - y2[i]);
        preCoef[totalAN*23+i] = -0.48923829943525*y[i]*(3.0*x2[i] - y2[i])*(x2[i] + y2[i] - 8.0*z2[i]);
        preCoef[totalAN*24+i] = 4.79353678497332*x[i]*y[i]*z[i]*(-x2[i] - y2[i] + 2.0*z2[i]);
        preCoef[totalAN*25+i] = 0.452946651195697*y[i]*(21.0*z4[i] - 14.0*z2[i]*r2[i] + r4[i]);
        preCoef[totalAN*26+i] = 0.116950322453424*z[i]*(63.0*z4[i] - 70.0*z2[i]*r2[i] + 15.0*r4[i]);
        preCoef[totalAN*27+i] = 0.452946651195697*x[i]*(21.0*z4[i] - 14.0*z2[i]*r2[i] + r4[i]);
        preCoef[totalAN*28+i] = 2.39676839248666*z[i]*(x2[i] - y2[i])*(-x2[i] - y2[i] + 2.0*z2[i]);
        preCoef[totalAN*29+i] = -0.48923829943525*x[i]*(x2[i] - 3.0*y2[i])*(x2[i] + y2[i] - 8.0*z2[i]);
        preCoef[totalAN*30+i] = 2.07566231488104*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
        preCoef[totalAN*31+i] = 0.65638205684017*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);

        if(return_derivatives){
        prCofDX[totalAN*21+i] = 13.1276411368034*x[i]*y[i]*(x2[i] - y2[i]);
        prCofDX[totalAN*22+i] = 8.30264925952416*y[i]*z[i]*(3.0*x2[i] - y2[i]);
        prCofDX[totalAN*23+i] = 1.956953197741*x[i]*y[i]*(-3.0*x2[i] - y2[i] + 12.0*z2[i]);
        prCofDX[totalAN*24+i] = 4.79353678497332*y[i]*z[i]*(-3.0*x2[i] - y2[i] + 2.0*z2[i]);
        prCofDX[totalAN*25+i] = 1.81178660478279*x[i]*y[i]*(x2[i] + y2[i] - 6.0*z2[i]);
        prCofDX[totalAN*26+i] = 2.33900644906847*x[i]*z[i]*(3.0*x2[i] + 3.0*y2[i] - 4.0*z2[i]);
        prCofDX[totalAN*27+i] = 2.26473325597848*x4[i] + 2.71767990717418*x2[i]*y2[i] - 16.3060794430451*x2[i]*z2[i] + 0.452946651195697*y4[i] - 5.43535981434836*y2[i]*z2[i] + 3.62357320956558*z4[i];
        prCofDX[totalAN*28+i] = 9.58707356994665*x[i]*z[i]*(-x2[i] + z2[i]);
        prCofDX[totalAN*29+i] = -0.978476598870501*x2[i]*(x2[i] - 3.0*y2[i]) - 1.46771489830575*(x2[i] - y2[i])*(x2[i] + y2[i] - 8.0*z2[i]);
        prCofDX[totalAN*30+i] = 8.30264925952416*x[i]*z[i]*(x2[i] - 3.0*y2[i]);
        prCofDX[totalAN*31+i] = 3.28191028420085*x4[i] - 19.6914617052051*x2[i]*y2[i] + 3.28191028420085*y4[i];

        prCofDY[totalAN*21+i] = 3.28191028420085*x4[i] - 19.6914617052051*x2[i]*y2[i] + 3.28191028420085*y4[i];
        prCofDY[totalAN*22+i] = 8.30264925952416*x[i]*z[i]*(x2[i] - 3.0*y2[i]);
        prCofDY[totalAN*23+i] = -0.978476598870501*y2[i]*(3.0*x2[i] - y2[i]) - 1.46771489830575*(x2[i] - y2[i])*(x2[i] + y2[i] - 8.0*z2[i]);
        prCofDY[totalAN*24+i] = 4.79353678497332*x[i]*z[i]*(-x2[i] - 3.0*y2[i] + 2.0*z2[i]);
        prCofDY[totalAN*25+i] = 0.452946651195697*x4[i] + 2.71767990717418*x2[i]*y2[i] - 5.43535981434836*x2[i]*z2[i] + 2.26473325597848*y4[i] - 16.3060794430451*y2[i]*z2[i] + 3.62357320956558*z4[i];
        prCofDY[totalAN*26+i] = 2.33900644906847*y[i]*z[i]*(3.0*x2[i] + 3.0*y2[i] - 4.0*z2[i]);
        prCofDY[totalAN*27+i] = 1.81178660478279*x[i]*y[i]*(x2[i] + y2[i] - 6.0*z2[i]);
        prCofDY[totalAN*28+i] = 9.58707356994665*y[i]*z[i]*(y2[i] - z2[i]);
        prCofDY[totalAN*29+i] = 1.956953197741*x[i]*y[i]*(x2[i] + 3.0*y2[i] - 12.0*z2[i]);
        prCofDY[totalAN*30+i] = -8.30264925952416*y[i]*z[i]*(3.0*x2[i] - y2[i]);
        prCofDY[totalAN*31+i] = 13.1276411368034*x[i]*y[i]*(-x2[i] + y2[i]);

        prCofDZ[totalAN*21+i] = 0;
        prCofDZ[totalAN*22+i] = 8.30264925952416*x[i]*y[i]*(x2[i] - y2[i]);
        prCofDZ[totalAN*23+i] = 7.82781279096401*y[i]*z[i]*(3.0*x2[i] - y2[i]);
        prCofDZ[totalAN*24+i] = -4.79353678497332*x[i]*y[i]*(x2[i] + y2[i] - 6.0*z2[i]);
        prCofDZ[totalAN*25+i] = 3.62357320956558*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 4.0*z2[i]);
        prCofDZ[totalAN*26+i] = 20.4663064293491*z4[i] - 17.5425483680135*z2[i]*r2[i] + 1.75425483680135*r4[i];
        prCofDZ[totalAN*27+i] = 3.62357320956558*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 4.0*z2[i]);
        prCofDZ[totalAN*28+i] = -2.39676839248666*(x2[i] - y2[i])*(x2[i] + y2[i] - 6.0*z2[i]);
        prCofDZ[totalAN*29+i] = 7.82781279096401*x[i]*z[i]*(x2[i] - 3.0*y2[i]);
        prCofDZ[totalAN*30+i] = 2.07566231488104*x4[i] - 12.4539738892862*x2[i]*y2[i] + 2.07566231488104*y4[i];
        prCofDZ[totalAN*31+i] = 0;
        }
    if (lMax > 5){
        preCoef[totalAN*32+i] = 1.36636821038383*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
        preCoef[totalAN*33+i] = 2.36661916223175*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
        preCoef[totalAN*34+i] = -2.0182596029149*x[i]*y[i]*(x2[i] - y2[i])*(x2[i] + y2[i] - 10.0*z2[i]);
        preCoef[totalAN*35+i] = 0.921205259514923*y[i]*z[i]*(3.0*x2[i] - y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 8.0*z2[i]);
        preCoef[totalAN*36+i] = 0.921205259514923*x[i]*y[i]*(33.0*z4[i] - 18.0*z2[i]*r2[i] + r4[i]);
        preCoef[totalAN*37+i] = 0.582621362518731*y[i]*z[i]*(33.0*z4[i] - 30.0*z2[i]*r2[i] + 5.0*r4[i]);
        preCoef[totalAN*38+i] = 14.6844857238222*z6[i] - 20.024298714303*z4[i]*r2[i] + 6.67476623810098*z2[i]*r4[i] - 0.317846011338142*r6[i];
        preCoef[totalAN*39+i] = 0.582621362518731*x[i]*z[i]*(33.0*z4[i] - 30.0*z2[i]*r2[i] + 5.0*r4[i]);
        preCoef[totalAN*40+i] = 0.460602629757462*(x2[i] - y2[i])*(33.0*z4[i] - 18.0*z2[i]*r2[i] + r4[i]);
        preCoef[totalAN*41+i] = 0.921205259514923*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 8.0*z2[i]);
        preCoef[totalAN*42+i] = -0.504564900728724*(x2[i] + y2[i] - 10.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
        preCoef[totalAN*43+i] = 2.36661916223175*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
        preCoef[totalAN*44+i] = 0.683184105191914*x6[i] - 10.2477615778787*x4[i]*y2[i] + 10.2477615778787*x2[i]*y4[i] - 0.683184105191914*y6[i];

        if(return_derivatives){
        prCofDX[totalAN*32+i] = 4.09910463115149*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
        prCofDX[totalAN*33+i] = 47.332383244635*x[i]*y[i]*z[i]*(x2[i] - y2[i]);
        prCofDX[totalAN*34+i] = -2.0182596029149*y[i]*(5.0*x4[i] - 30.0*x2[i]*z2[i] - y4[i] + 10.0*y2[i]*z2[i]);
        prCofDX[totalAN*35+i] = 11.0544631141791*x[i]*y[i]*z[i]*(-3.0*x2[i] - y2[i] + 4.0*z2[i]);
        prCofDX[totalAN*36+i] = 0.921205259514923*y[i]*(5.0*x4[i] + 6.0*x2[i]*y2[i] - 48.0*x2[i]*z2[i] + y4[i] - 16.0*y2[i]*z2[i] + 16.0*z4[i]);
        prCofDX[totalAN*37+i] = 11.6524272503746*x[i]*y[i]*z[i]*(x2[i] + y2[i] - 2.0*z2[i]);
        prCofDX[totalAN*38+i] = 1.90707606802885*x[i]*(-21.0*z4[i] + 14.0*z2[i]*r2[i] - r4[i]);
        prCofDX[totalAN*39+i] = 0.582621362518731*z[i]*(25.0*x4[i] + 30.0*x2[i]*y2[i] - 60.0*x2[i]*z2[i] + 5.0*y4[i] - 20.0*y2[i]*z2[i] + 8.0*z4[i]);
        prCofDX[totalAN*40+i] = 0.921205259514923*x[i]*(3.0*x4[i] + 2.0*x2[i]*y2[i] - 32.0*x2[i]*z2[i] - y4[i] + 16.0*z4[i]);
        prCofDX[totalAN*41+i] = 2.76361577854477*z[i]*(-2.0*x2[i]*(x2[i] - 3.0*y2[i]) + (x2[i] - y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 8.0*z2[i]));
        prCofDX[totalAN*42+i] = 1.00912980145745*x[i]*(-x4[i] + 6.0*x2[i]*y2[i] - y4[i] - 2.0*(x2[i] - 3.0*y2[i])*(x2[i] + y2[i] - 10.0*z2[i]));
        prCofDX[totalAN*43+i] = 11.8330958111588*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
        prCofDX[totalAN*44+i] = 4.09910463115149*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);

        prCofDY[totalAN*32+i] = 4.09910463115149*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
        prCofDY[totalAN*33+i] = 11.8330958111588*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
        prCofDY[totalAN*34+i] = -2.0182596029149*x[i]*(x4[i] - 10.0*x2[i]*z2[i] - 5.0*y4[i] + 30.0*y2[i]*z2[i]);
        prCofDY[totalAN*35+i] = 2.76361577854477*z[i]*(-2.0*y2[i]*(3.0*x2[i] - y2[i]) + (x2[i] - y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 8.0*z2[i]));
        prCofDY[totalAN*36+i] = 0.921205259514923*x[i]*(x4[i] + 6.0*x2[i]*y2[i] - 16.0*x2[i]*z2[i] + 5.0*y4[i] - 48.0*y2[i]*z2[i] + 16.0*z4[i]);
        prCofDY[totalAN*37+i] = 0.582621362518731*z[i]*(5.0*x4[i] + 30.0*x2[i]*y2[i] - 20.0*x2[i]*z2[i] + 25.0*y4[i] - 60.0*y2[i]*z2[i] + 8.0*z4[i]);
        prCofDY[totalAN*38+i] = 1.90707606802885*y[i]*(-21.0*z4[i] + 14.0*z2[i]*r2[i] - r4[i]);
        prCofDY[totalAN*39+i] = 11.6524272503746*x[i]*y[i]*z[i]*(x2[i] + y2[i] - 2.0*z2[i]);
        prCofDY[totalAN*40+i] = 0.921205259514923*y[i]*(x4[i] - 2.0*x2[i]*y2[i] - 3.0*y4[i] + 32.0*y2[i]*z2[i] - 16.0*z4[i]);
        prCofDY[totalAN*41+i] = 11.0544631141791*x[i]*y[i]*z[i]*(x2[i] + 3.0*y2[i] - 4.0*z2[i]);
        prCofDY[totalAN*42+i] = 1.00912980145745*y[i]*(5.0*x4[i] + 10.0*x2[i]*y2[i] - 60.0*x2[i]*z2[i] - 3.0*y4[i] + 20.0*y2[i]*z2[i]);
        prCofDY[totalAN*43+i] = -47.332383244635*x[i]*y[i]*z[i]*(x2[i] - y2[i]);
        prCofDY[totalAN*44+i] = 4.09910463115149*y[i]*(-5.0*x4[i] + 10.0*x2[i]*y2[i] - y4[i]);

        prCofDZ[totalAN*32+i] = 0;
        prCofDZ[totalAN*33+i] = 2.36661916223175*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
        prCofDZ[totalAN*34+i] = 40.3651920582979*x[i]*y[i]*z[i]*(x2[i] - y2[i]);
        prCofDZ[totalAN*35+i] = -2.76361577854477*y[i]*(3.0*x2[i] - y2[i])*(x2[i] + y2[i] - 8.0*z2[i]);
        prCofDZ[totalAN*36+i] = 29.4785683044776*x[i]*y[i]*z[i]*(-x2[i] - y2[i] + 2.0*z2[i]);
        prCofDZ[totalAN*37+i] = 2.91310681259366*y[i]*(21.0*z4[i] - 14.0*z2[i]*r2[i] + r4[i]);
        prCofDZ[totalAN*38+i] = 0.762830427211541*z[i]*(63.0*z4[i] - 70.0*z2[i]*r2[i] + 15.0*r4[i]);
        prCofDZ[totalAN*39+i] = 2.91310681259366*x[i]*(21.0*z4[i] - 14.0*z2[i]*r2[i] + r4[i]);
        prCofDZ[totalAN*40+i] = 14.7392841522388*z[i]*(x2[i] - y2[i])*(-x2[i] - y2[i] + 2.0*z2[i]);
        prCofDZ[totalAN*41+i] = -2.76361577854477*x[i]*(x2[i] - 3.0*y2[i])*(x2[i] + y2[i] - 8.0*z2[i]);
        prCofDZ[totalAN*42+i] = 10.0912980145745*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
        prCofDZ[totalAN*43+i] = 2.36661916223175*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
        prCofDZ[totalAN*44+i] = 0;
        }
    if (lMax > 6){
        preCoef[totalAN*45+i] = 0.707162732524596*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
        preCoef[totalAN*46+i] = 5.2919213236038*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
        preCoef[totalAN*47+i] = -0.51891557872026*y[i]*(x2[i] + y2[i] - 12.0*z2[i])*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
        preCoef[totalAN*48+i] = 4.15132462976208*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 10.0*z2[i]);
        preCoef[totalAN*49+i] = 0.156458933862294*y[i]*(3.0*x2[i] - y2[i])*(143.0*z4[i] - 66.0*z2[i]*r2[i] + 3.0*r4[i]);
        preCoef[totalAN*50+i] = 0.442532692444983*x[i]*y[i]*z[i]*(143.0*z4[i] - 110.0*z2[i]*r2[i] + 15.0*r4[i]);
        preCoef[totalAN*51+i] = 0.0903316075825173*y[i]*(429.0*z6[i] - 495.0*z4[i]*r2[i] + 135.0*z2[i]*r4[i] - 5.0*r6[i]);
        preCoef[totalAN*52+i] = 0.0682842769120049*z[i]*(429.0*z6[i] - 693.0*z4[i]*r2[i] + 315.0*z2[i]*r4[i] - 35.0*r6[i]);
        preCoef[totalAN*53+i] = 0.0903316075825173*x[i]*(429.0*z6[i] - 495.0*z4[i]*r2[i] + 135.0*z2[i]*r4[i] - 5.0*r6[i]);
        preCoef[totalAN*54+i] = 0.221266346222491*z[i]*(x2[i] - y2[i])*(143.0*z4[i] - 110.0*z2[i]*r2[i] + 15.0*r4[i]);
        preCoef[totalAN*55+i] = 0.156458933862294*x[i]*(x2[i] - 3.0*y2[i])*(143.0*z4[i] - 66.0*z2[i]*r2[i] + 3.0*r4[i]);
        preCoef[totalAN*56+i] = 1.03783115744052*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 10.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
        preCoef[totalAN*57+i] = -0.51891557872026*x[i]*(x2[i] + y2[i] - 12.0*z2[i])*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
        preCoef[totalAN*58+i] = 2.6459606618019*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
        preCoef[totalAN*59+i] = 0.707162732524596*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);

        if(return_derivatives){
        prCofDX[totalAN*45+i] = 9.90027825534435*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
        prCofDX[totalAN*46+i] = 15.8757639708114*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
        prCofDX[totalAN*47+i] = 1.03783115744052*x[i]*y[i]*(-5.0*x4[i] + 10.0*x2[i]*y2[i] - y4[i] - 10.0*(x2[i] - y2[i])*(x2[i] + y2[i] - 12.0*z2[i]));
        prCofDX[totalAN*48+i] = -4.15132462976208*y[i]*z[i]*(15.0*x4[i] - 30.0*x2[i]*z2[i] - 3.0*y4[i] + 10.0*y2[i]*z2[i]);
        prCofDX[totalAN*49+i] = 0.938753603173764*x[i]*y[i]*(9.0*x4[i] + 10.0*x2[i]*y2[i] - 120.0*x2[i]*z2[i] + y4[i] - 40.0*y2[i]*z2[i] + 80.0*z4[i]);
        prCofDX[totalAN*50+i] = 0.442532692444983*y[i]*z[i]*(75.0*x4[i] + 90.0*x2[i]*y2[i] - 240.0*x2[i]*z2[i] + 15.0*y4[i] - 80.0*y2[i]*z2[i] + 48.0*z4[i]);
        prCofDX[totalAN*51+i] = -2.70994822747552*x[i]*y[i]*(33.0*z4[i] - 18.0*z2[i]*r2[i] + r4[i]);
        prCofDX[totalAN*52+i] = 2.86793963030421*x[i]*z[i]*(-33.0*z4[i] + 30.0*z2[i]*r2[i] - 5.0*r4[i]);
        prCofDX[totalAN*53+i] = -3.16160626538811*x6[i] - 6.7748705686888*x4[i]*y2[i] + 54.1989645495104*x4[i]*z2[i] - 4.06492234121328*x2[i]*y4[i] + 65.0387574594125*x2[i]*y2[i]*z2[i] - 65.0387574594125*x2[i]*z4[i] - 0.451658037912587*y6[i] + 10.8397929099021*y4[i]*z2[i] - 21.6795858198042*y2[i]*z4[i] + 5.78122288528111*z6[i];
        prCofDX[totalAN*54+i] = 0.442532692444983*x[i]*z[i]*(45.0*x4[i] + 30.0*x2[i]*y2[i] - 160.0*x2[i]*z2[i] - 15.0*y4[i] + 48.0*z4[i]);
        prCofDX[totalAN*55+i] = -1.87750720634753*x2[i]*(x2[i] - 3.0*y2[i])*(-x2[i] - y2[i] + 10.0*z2[i]) + 0.469376801586882*(x2[i] - y2[i])*(143.0*z4[i] - 66.0*z2[i]*r2[i] + 3.0*r4[i]);
        prCofDX[totalAN*56+i] = 2.07566231488104*x[i]*z[i]*(-9.0*x4[i] + 30.0*x2[i]*y2[i] + 20.0*x2[i]*z2[i] + 15.0*y4[i] - 60.0*y2[i]*z2[i]);
        prCofDX[totalAN*57+i] = -1.03783115744052*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]) - 2.5945778936013*(x2[i] + y2[i] - 12.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
        prCofDX[totalAN*58+i] = 15.8757639708114*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
        prCofDX[totalAN*59+i] = 4.95013912767217*x6[i] - 74.2520869150826*x4[i]*y2[i] + 74.2520869150826*x2[i]*y4[i] - 4.95013912767217*y6[i];

        prCofDY[totalAN*45+i] = 4.95013912767217*x6[i] - 74.2520869150826*x4[i]*y2[i] + 74.2520869150826*x2[i]*y4[i] - 4.95013912767217*y6[i];
        prCofDY[totalAN*46+i] = 15.8757639708114*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
        prCofDY[totalAN*47+i] = -1.03783115744052*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]) - 2.5945778936013*(x2[i] + y2[i] - 12.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
        prCofDY[totalAN*48+i] = -4.15132462976208*x[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*z2[i] - 15.0*y4[i] + 30.0*y2[i]*z2[i]);
        prCofDY[totalAN*49+i] = -1.87750720634753*y2[i]*(3.0*x2[i] - y2[i])*(-x2[i] - y2[i] + 10.0*z2[i]) + 0.469376801586882*(x2[i] - y2[i])*(143.0*z4[i] - 66.0*z2[i]*r2[i] + 3.0*r4[i]);
        prCofDY[totalAN*50+i] = 0.442532692444983*x[i]*z[i]*(15.0*x4[i] + 90.0*x2[i]*y2[i] - 80.0*x2[i]*z2[i] + 75.0*y4[i] - 240.0*y2[i]*z2[i] + 48.0*z4[i]);
        prCofDY[totalAN*51+i] = -0.451658037912587*x6[i] - 4.06492234121328*x4[i]*y2[i] + 10.8397929099021*x4[i]*z2[i] - 6.7748705686888*x2[i]*y4[i] + 65.0387574594125*x2[i]*y2[i]*z2[i] - 21.6795858198042*x2[i]*z4[i] - 3.16160626538811*y6[i] + 54.1989645495104*y4[i]*z2[i] - 65.0387574594125*y2[i]*z4[i] + 5.78122288528111*z6[i];
        prCofDY[totalAN*52+i] = 2.86793963030421*y[i]*z[i]*(-33.0*z4[i] + 30.0*z2[i]*r2[i] - 5.0*r4[i]);
        prCofDY[totalAN*53+i] = -2.70994822747552*x[i]*y[i]*(33.0*z4[i] - 18.0*z2[i]*r2[i] + r4[i]);
        prCofDY[totalAN*54+i] = 0.442532692444983*y[i]*z[i]*(15.0*x4[i] - 30.0*x2[i]*y2[i] - 45.0*y4[i] + 160.0*y2[i]*z2[i] - 48.0*z4[i]);
        prCofDY[totalAN*55+i] = -0.938753603173764*x[i]*y[i]*(x4[i] + 10.0*x2[i]*y2[i] - 40.0*x2[i]*z2[i] + 9.0*y4[i] - 120.0*y2[i]*z2[i] + 80.0*z4[i]);
        prCofDY[totalAN*56+i] = 2.07566231488104*y[i]*z[i]*(15.0*x4[i] + 30.0*x2[i]*y2[i] - 60.0*x2[i]*z2[i] - 9.0*y4[i] + 20.0*y2[i]*z2[i]);
        prCofDY[totalAN*57+i] = 1.03783115744052*x[i]*y[i]*(9.0*x4[i] + 10.0*x2[i]*y2[i] - 120.0*x2[i]*z2[i] - 15.0*y4[i] + 120.0*y2[i]*z2[i]);
        prCofDY[totalAN*58+i] = -15.8757639708114*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
        prCofDY[totalAN*59+i] = 9.90027825534435*x[i]*y[i]*(-3.0*x4[i] + 10.0*x2[i]*y2[i] - 3.0*y4[i]);

        prCofDZ[totalAN*45+i] = 0;
        prCofDZ[totalAN*46+i] = 5.2919213236038*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
        prCofDZ[totalAN*47+i] = 12.4539738892862*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
        prCofDZ[totalAN*48+i] = -12.4539738892862*x[i]*y[i]*(x2[i] - y2[i])*(x2[i] + y2[i] - 10.0*z2[i]);
        prCofDZ[totalAN*49+i] = 6.25835735449176*y[i]*z[i]*(3.0*x2[i] - y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 8.0*z2[i]);
        prCofDZ[totalAN*50+i] = 6.63799038667474*x[i]*y[i]*(33.0*z4[i] - 18.0*z2[i]*r2[i] + r4[i]);
        prCofDZ[totalAN*51+i] = 4.33591716396083*y[i]*z[i]*(33.0*z4[i] - 30.0*z2[i]*r2[i] + 5.0*r4[i]);
        prCofDZ[totalAN*52+i] = 110.415675766712*z6[i] - 150.566830590971*z4[i]*r2[i] + 50.1889435303236*z2[i]*r4[i] - 2.38994969192017*r6[i];
        prCofDZ[totalAN*53+i] = 4.33591716396083*x[i]*z[i]*(33.0*z4[i] - 30.0*z2[i]*r2[i] + 5.0*r4[i]);
        prCofDZ[totalAN*54+i] = 3.31899519333737*(x2[i] - y2[i])*(33.0*z4[i] - 18.0*z2[i]*r2[i] + r4[i]);
        prCofDZ[totalAN*55+i] = 6.25835735449176*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 8.0*z2[i]);
        prCofDZ[totalAN*56+i] = -3.11349347232156*(x2[i] + y2[i] - 10.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
        prCofDZ[totalAN*57+i] = 12.4539738892862*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
        prCofDZ[totalAN*58+i] = 2.6459606618019*x6[i] - 39.6894099270285*x4[i]*y2[i] + 39.6894099270285*x2[i]*y4[i] - 2.6459606618019*y6[i];
        prCofDZ[totalAN*59+i] = 0;
        }
    if (lMax > 7){
        preCoef[totalAN*60+i] = 5.83141328139864*x[i]*y[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
        preCoef[totalAN*61+i] = 2.91570664069932*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
        preCoef[totalAN*62+i] = -1.06466553211909*x[i]*y[i]*(x2[i] + y2[i] - 14.0*z2[i])*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
        preCoef[totalAN*63+i] = 3.44991062209811*y[i]*z[i]*(-x2[i] - y2[i] + 4.0*z2[i])*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
        preCoef[totalAN*64+i] = 1.91366609903732*x[i]*y[i]*(x2[i] - y2[i])*(65.0*z4[i] - 26.0*z2[i]*r2[i] + r4[i]);
        preCoef[totalAN*65+i] = 1.23526615529554*y[i]*z[i]*(3.0*x2[i] - y2[i])*(39.0*z4[i] - 26.0*z2[i]*r2[i] + 3.0*r4[i]);
        preCoef[totalAN*66+i] = 0.912304516869819*x[i]*y[i]*(143.0*z6[i] - 143.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i]);
        preCoef[totalAN*67+i] = 0.10904124589878*y[i]*z[i]*(715.0*z6[i] - 1001.0*z4[i]*r2[i] + 385.0*z2[i]*r4[i] - 35.0*r6[i]);
        preCoef[totalAN*68+i] = 58.4733681132208*z8[i] - 109.150287144679*z6[i]*r2[i] + 62.9713195065454*z4[i]*r4[i] - 11.4493308193719*z2[i]*r6[i] + 0.318036967204775*r8[i];
        preCoef[totalAN*69+i] = 0.10904124589878*x[i]*z[i]*(715.0*z6[i] - 1001.0*z4[i]*r2[i] + 385.0*z2[i]*r4[i] - 35.0*r6[i]);
        preCoef[totalAN*70+i] = 0.456152258434909*(x2[i] - y2[i])*(143.0*z6[i] - 143.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i]);
        preCoef[totalAN*71+i] = 1.23526615529554*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(39.0*z4[i] - 26.0*z2[i]*r2[i] + 3.0*r4[i]);
        preCoef[totalAN*72+i] = 0.478416524759331*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(65.0*z4[i] - 26.0*z2[i]*r2[i] + r4[i]);
        preCoef[totalAN*73+i] = 3.44991062209811*x[i]*z[i]*(-x2[i] - y2[i] + 4.0*z2[i])*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
        preCoef[totalAN*74+i] = -0.532332766059543*(x2[i] + y2[i] - 14.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
        preCoef[totalAN*75+i] = 2.91570664069932*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
        preCoef[totalAN*76+i] = 0.72892666017483*x8[i] - 20.4099464848952*x6[i]*y2[i] + 51.0248662122381*x4[i]*y4[i] - 20.4099464848952*x2[i]*y6[i] + 0.72892666017483*y8[i];

        if(return_derivatives){
        prCofDX[totalAN*60+i] = 5.83141328139864*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
        prCofDX[totalAN*61+i] = 40.8198929697905*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
        prCofDX[totalAN*62+i] = 1.06466553211909*y[i]*(x2[i]*(-6.0*x4[i] + 20.0*x2[i]*y2[i] - 6.0*y4[i]) - 3.0*(x2[i] + y2[i] - 14.0*z2[i])*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]));
        prCofDX[totalAN*63+i] = 6.89982124419622*x[i]*y[i]*z[i]*(-5.0*x4[i] + 10.0*x2[i]*y2[i] - y4[i] + 10.0*(x2[i] - y2[i])*(-x2[i] - y2[i] + 4.0*z2[i]));
        prCofDX[totalAN*64+i] = 1.91366609903732*y[i]*(7.0*x6[i] + 5.0*x4[i]*y2[i] - 120.0*x4[i]*z2[i] - 3.0*x2[i]*y4[i] + 120.0*x2[i]*z4[i] - y6[i] + 24.0*y4[i]*z2[i] - 40.0*y2[i]*z4[i]);
        prCofDX[totalAN*65+i] = 2.47053231059109*x[i]*y[i]*z[i]*(27.0*x4[i] + 30.0*x2[i]*y2[i] - 120.0*x2[i]*z2[i] + 3.0*y4[i] - 40.0*y2[i]*z2[i] + 48.0*z4[i]);
        prCofDX[totalAN*66+i] = -0.912304516869819*y[i]*(7.0*x6[i] + 15.0*x4[i]*y2[i] - 150.0*x4[i]*z2[i] + 9.0*x2[i]*y4[i] - 180.0*x2[i]*y2[i]*z2[i] + 240.0*x2[i]*z4[i] + y6[i] - 30.0*y4[i]*z2[i] + 80.0*y2[i]*z4[i] - 32.0*z6[i]);
        prCofDX[totalAN*67+i] = -1.52657744258292*x[i]*y[i]*z[i]*(143.0*z4[i] - 110.0*z2[i]*r2[i] + 15.0*r4[i]);
        prCofDX[totalAN*68+i] = 0.50885914752764*x[i]*(-429.0*z6[i] + 495.0*z4[i]*r2[i] - 135.0*z2[i]*r4[i] + 5.0*r6[i]);
        prCofDX[totalAN*69+i] = -0.10904124589878*z[i]*(245.0*x6[i] + 525.0*x4[i]*y2[i] - 1400.0*x4[i]*z2[i] + 315.0*x2[i]*y4[i] - 1680.0*x2[i]*y2[i]*z2[i] + 1008.0*x2[i]*z4[i] + 35.0*y6[i] - 280.0*y4[i]*z2[i] + 336.0*y2[i]*z4[i] - 64.0*z6[i]);
        prCofDX[totalAN*70+i] = -1.82460903373964*x[i]*(2.0*x6[i] + 3.0*x4[i]*y2[i] - 45.0*x4[i]*z2[i] - 30.0*x2[i]*y2[i]*z2[i] + 80.0*x2[i]*z4[i] - y6[i] + 15.0*y4[i]*z2[i] - 16.0*z6[i]);
        prCofDX[totalAN*71+i] = 1.23526615529554*z[i]*(-4.0*x2[i]*(x2[i] - 3.0*y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 10.0*z2[i]) + 3.0*(x2[i] - y2[i])*(39.0*z4[i] - 26.0*z2[i]*r2[i] + 3.0*r4[i]));
        prCofDX[totalAN*72+i] = 1.91366609903732*x[i]*((x2[i] - 3.0*y2[i])*(65.0*z4[i] - 26.0*z2[i]*r2[i] + r4[i]) - (-x2[i] - y2[i] + 12.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]));
        prCofDX[totalAN*73+i] = 3.44991062209811*z[i]*(-2.0*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]) + 5.0*(-x2[i] - y2[i] + 4.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]));
        prCofDX[totalAN*74+i] = -2.12933106423817*x[i]*(2.0*x6[i] - 21.0*x4[i]*y2[i] - 21.0*x4[i]*z2[i] + 210.0*x2[i]*y2[i]*z2[i] + 7.0*y6[i] - 105.0*y4[i]*z2[i]);
        prCofDX[totalAN*75+i] = 20.4099464848952*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
        prCofDX[totalAN*76+i] = 5.83141328139864*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);

        prCofDY[totalAN*60+i] = 5.83141328139864*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
        prCofDY[totalAN*61+i] = 20.4099464848952*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
        prCofDY[totalAN*62+i] = 1.06466553211909*x[i]*(y2[i]*(-6.0*x4[i] + 20.0*x2[i]*y2[i] - 6.0*y4[i]) - 3.0*(x2[i] + y2[i] - 14.0*z2[i])*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]));
        prCofDY[totalAN*63+i] = 3.44991062209811*z[i]*(-2.0*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]) + 5.0*(-x2[i] - y2[i] + 4.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]));
        prCofDY[totalAN*64+i] = 1.91366609903732*x[i]*(x6[i] + 3.0*x4[i]*y2[i] - 24.0*x4[i]*z2[i] - 5.0*x2[i]*y4[i] + 40.0*x2[i]*z4[i] - 7.0*y6[i] + 120.0*y4[i]*z2[i] - 120.0*y2[i]*z4[i]);
        prCofDY[totalAN*65+i] = 1.23526615529554*z[i]*(-4.0*y2[i]*(3.0*x2[i] - y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 10.0*z2[i]) + 3.0*(x2[i] - y2[i])*(39.0*z4[i] - 26.0*z2[i]*r2[i] + 3.0*r4[i]));
        prCofDY[totalAN*66+i] = -0.912304516869819*x[i]*(x6[i] + 9.0*x4[i]*y2[i] - 30.0*x4[i]*z2[i] + 15.0*x2[i]*y4[i] - 180.0*x2[i]*y2[i]*z2[i] + 80.0*x2[i]*z4[i] + 7.0*y6[i] - 150.0*y4[i]*z2[i] + 240.0*y2[i]*z4[i] - 32.0*z6[i]);
        prCofDY[totalAN*67+i] = -0.10904124589878*z[i]*(35.0*x6[i] + 315.0*x4[i]*y2[i] - 280.0*x4[i]*z2[i] + 525.0*x2[i]*y4[i] - 1680.0*x2[i]*y2[i]*z2[i] + 336.0*x2[i]*z4[i] + 245.0*y6[i] - 1400.0*y4[i]*z2[i] + 1008.0*y2[i]*z4[i] - 64.0*z6[i]);
        prCofDY[totalAN*68+i] = 0.50885914752764*y[i]*(-429.0*z6[i] + 495.0*z4[i]*r2[i] - 135.0*z2[i]*r4[i] + 5.0*r6[i]);
        prCofDY[totalAN*69+i] = -1.52657744258292*x[i]*y[i]*z[i]*(143.0*z4[i] - 110.0*z2[i]*r2[i] + 15.0*r4[i]);
        prCofDY[totalAN*70+i] = -1.82460903373964*y[i]*(x6[i] - 15.0*x4[i]*z2[i] - 3.0*x2[i]*y4[i] + 30.0*x2[i]*y2[i]*z2[i] - 2.0*y6[i] + 45.0*y4[i]*z2[i] - 80.0*y2[i]*z4[i] + 16.0*z6[i]);
        prCofDY[totalAN*71+i] = -2.47053231059109*x[i]*y[i]*z[i]*(3.0*x4[i] + 30.0*x2[i]*y2[i] - 40.0*x2[i]*z2[i] + 27.0*y4[i] - 120.0*y2[i]*z2[i] + 48.0*z4[i]);
        prCofDY[totalAN*72+i] = -1.91366609903732*y[i]*((3.0*x2[i] - y2[i])*(65.0*z4[i] - 26.0*z2[i]*r2[i] + r4[i]) - (x2[i] + y2[i] - 12.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]));
        prCofDY[totalAN*73+i] = 6.89982124419622*x[i]*y[i]*z[i]*(9.0*x4[i] + 10.0*x2[i]*y2[i] - 40.0*x2[i]*z2[i] - 15.0*y4[i] + 40.0*y2[i]*z2[i]);
        prCofDY[totalAN*74+i] = 2.12933106423817*y[i]*(7.0*x6[i] - 105.0*x4[i]*z2[i] - 21.0*x2[i]*y4[i] + 210.0*x2[i]*y2[i]*z2[i] + 2.0*y6[i] - 21.0*y4[i]*z2[i]);
        prCofDY[totalAN*75+i] = -40.8198929697905*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
        prCofDY[totalAN*76+i] = 5.83141328139864*y[i]*(-7.0*x6[i] + 35.0*x4[i]*y2[i] - 21.0*x2[i]*y4[i] + y6[i]);

        prCofDZ[totalAN*60+i] = 0;
        prCofDZ[totalAN*61+i] = 2.91570664069932*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
        prCofDZ[totalAN*62+i] = 29.8106348993344*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
        prCofDZ[totalAN*63+i] = -3.44991062209811*y[i]*(x2[i] + y2[i] - 12.0*z2[i])*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
        prCofDZ[totalAN*64+i] = 30.6186575845972*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(-3.0*x2[i] - 3.0*y2[i] + 10.0*z2[i]);
        prCofDZ[totalAN*65+i] = 1.23526615529554*y[i]*(3.0*x2[i] - y2[i])*(143.0*z4[i] - 66.0*z2[i]*r2[i] + 3.0*r4[i]);
        prCofDZ[totalAN*66+i] = 3.64921806747928*x[i]*y[i]*z[i]*(143.0*z4[i] - 110.0*z2[i]*r2[i] + 15.0*r4[i]);
        prCofDZ[totalAN*67+i] = 0.76328872129146*y[i]*(429.0*z6[i] - 495.0*z4[i]*r2[i] + 135.0*z2[i]*r4[i] - 5.0*r6[i]);
        prCofDZ[totalAN*68+i] = 0.58155331146016*z[i]*(429.0*z6[i] - 693.0*z4[i]*r2[i] + 315.0*z2[i]*r4[i] - 35.0*r6[i]);
        prCofDZ[totalAN*69+i] = 0.76328872129146*x[i]*(429.0*z6[i] - 495.0*z4[i]*r2[i] + 135.0*z2[i]*r4[i] - 5.0*r6[i]);
        prCofDZ[totalAN*70+i] = 1.82460903373964*z[i]*(x2[i] - y2[i])*(143.0*z4[i] - 110.0*z2[i]*r2[i] + 15.0*r4[i]);
        prCofDZ[totalAN*71+i] = 1.23526615529554*x[i]*(x2[i] - 3.0*y2[i])*(143.0*z4[i] - 66.0*z2[i]*r2[i] + 3.0*r4[i]);
        prCofDZ[totalAN*72+i] = 7.65466439614929*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 10.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]);
        prCofDZ[totalAN*73+i] = -3.44991062209811*x[i]*(x2[i] + y2[i] - 12.0*z2[i])*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
        prCofDZ[totalAN*74+i] = 14.9053174496672*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
        prCofDZ[totalAN*75+i] = 2.91570664069932*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
        prCofDZ[totalAN*76+i] = 0;
        }
    if (lMax > 8){
        preCoef[totalAN*77+i] = 0.748900951853188*y[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
        preCoef[totalAN*78+i] = 25.4185411916376*x[i]*y[i]*z[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
        preCoef[totalAN*79+i] = -0.544905481344053*y[i]*(x2[i] + y2[i] - 16.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
        preCoef[totalAN*80+i] = 2.51681061069513*x[i]*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 14.0*z2[i])*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
        preCoef[totalAN*81+i] = 0.487378279039019*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(85.0*z4[i] - 30.0*z2[i]*r2[i] + r4[i]);
        preCoef[totalAN*82+i] = 16.3107969549167*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(17.0*z4[i] - 10.0*z2[i]*r2[i] + r4[i]);
        preCoef[totalAN*83+i] = 0.461708520016194*y[i]*(3.0*x2[i] - y2[i])*(221.0*z6[i] - 195.0*z4[i]*r2[i] + 39.0*z2[i]*r4[i] - r6[i]);
        preCoef[totalAN*84+i] = 1.209036709703*x[i]*y[i]*z[i]*(221.0*z6[i] - 273.0*z4[i]*r2[i] + 91.0*z2[i]*r4[i] - 7.0*r6[i]);
        preCoef[totalAN*85+i] = 0.0644418731522273*y[i]*(2431.0*z8[i] - 4004.0*z6[i]*r2[i] + 2002.0*z4[i]*r4[i] - 308.0*z2[i]*r6[i] + 7.0*r8[i]);
        preCoef[totalAN*86+i] = 0.00960642726438659*z[i]*(12155.0*z8[i] - 25740.0*z6[i]*r2[i] + 18018.0*z4[i]*r4[i] - 4620.0*z2[i]*r6[i] + 315.0*r8[i]);
        preCoef[totalAN*87+i] = 0.0644418731522273*x[i]*(2431.0*z8[i] - 4004.0*z6[i]*r2[i] + 2002.0*z4[i]*r4[i] - 308.0*z2[i]*r6[i] + 7.0*r8[i]);
        preCoef[totalAN*88+i] = 0.604518354851498*z[i]*(x2[i] - y2[i])*(221.0*z6[i] - 273.0*z4[i]*r2[i] + 91.0*z2[i]*r4[i] - 7.0*r6[i]);
        preCoef[totalAN*89+i] = 0.461708520016194*x[i]*(x2[i] - 3.0*y2[i])*(221.0*z6[i] - 195.0*z4[i]*r2[i] + 39.0*z2[i]*r4[i] - r6[i]);
        preCoef[totalAN*90+i] = 4.07769923872917*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(17.0*z4[i] - 10.0*z2[i]*r2[i] + r4[i]);
        preCoef[totalAN*91+i] = 0.487378279039019*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(85.0*z4[i] - 30.0*z2[i]*r2[i] + r4[i]);
        preCoef[totalAN*92+i] = 1.25840530534757*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 14.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
        preCoef[totalAN*93+i] = -0.544905481344053*x[i]*(x2[i] + y2[i] - 16.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
        preCoef[totalAN*94+i] = 3.1773176489547*z[i]*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
        preCoef[totalAN*95+i] = 0.748900951853188*x[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);

        if(return_derivatives){
        prCofDX[totalAN*77+i] = 53.9208685334296*x[i]*y[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
        prCofDX[totalAN*78+i] = 25.4185411916376*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
        prCofDX[totalAN*79+i] = 1.08981096268811*x[i]*y[i]*(-7.0*x6[i] + 35.0*x4[i]*y2[i] - 21.0*x2[i]*y4[i] + y6[i] - 7.0*(x2[i] + y2[i] - 16.0*z2[i])*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]));
        prCofDX[totalAN*80+i] = 7.5504318320854*y[i]*z[i]*(x2[i]*(-6.0*x4[i] + 20.0*x2[i]*y2[i] - 6.0*y4[i]) + (-3.0*x2[i] - 3.0*y2[i] + 14.0*z2[i])*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]));
        prCofDX[totalAN*81+i] = 3.89902623231215*x[i]*y[i]*(5.0*x6[i] - 105.0*x4[i]*z2[i] - 7.0*x2[i]*y4[i] + 70.0*x2[i]*y2[i]*z2[i] + 140.0*x2[i]*z4[i] - 2.0*y6[i] + 63.0*y4[i]*z2[i] - 140.0*y2[i]*z4[i]);
        prCofDX[totalAN*82+i] = 16.3107969549167*y[i]*z[i]*(7.0*x6[i] + 5.0*x4[i]*y2[i] - 40.0*x4[i]*z2[i] - 3.0*x2[i]*y4[i] + 24.0*x2[i]*z4[i] - y6[i] + 8.0*y4[i]*z2[i] - 8.0*y2[i]*z4[i]);
        prCofDX[totalAN*83+i] = -11.0810044803887*x[i]*y[i]*(x6[i] + 2.0*x4[i]*y2[i] - 27.0*x4[i]*z2[i] + x2[i]*y4[i] - 30.0*x2[i]*y2[i]*z2[i] + 60.0*x2[i]*z4[i] - 3.0*y4[i]*z2[i] + 20.0*y2[i]*z4[i] - 16.0*z6[i]);
        prCofDX[totalAN*84+i] = -1.209036709703*y[i]*z[i]*(49.0*x6[i] + 105.0*x4[i]*y2[i] - 350.0*x4[i]*z2[i] + 63.0*x2[i]*y4[i] - 420.0*x2[i]*y2[i]*z2[i] + 336.0*x2[i]*z4[i] + 7.0*y6[i] - 70.0*y4[i]*z2[i] + 112.0*y2[i]*z4[i] - 32.0*z6[i]);
        prCofDX[totalAN*85+i] = -3.60874489652473*x[i]*y[i]*(143.0*z6[i] - 143.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i]);
        prCofDX[totalAN*86+i] = 0.691662763035835*x[i]*z[i]*(-715.0*z6[i] + 1001.0*z4[i]*r2[i] - 385.0*z2[i]*r4[i] + 35.0*r6[i]);
        prCofDX[totalAN*87+i] = -3.60874489652473*x2[i]*(143.0*z6[i] - 143.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i]) + 156.658193633065*z8[i] - 258.025260101518*z6[i]*r2[i] + 129.012630050759*z4[i]*r4[i] - 19.848096930886*z2[i]*r6[i] + 0.451093112065591*r8[i];
        prCofDX[totalAN*88+i] = -2.41807341940599*x[i]*z[i]*(14.0*x6[i] + 21.0*x4[i]*y2[i] - 105.0*x4[i]*z2[i] - 70.0*x2[i]*y2[i]*z2[i] + 112.0*x2[i]*z4[i] - 7.0*y6[i] + 35.0*y4[i]*z2[i] - 16.0*z6[i]);
        prCofDX[totalAN*89+i] = -2.77025112009717*x2[i]*(x2[i] - 3.0*y2[i])*(65.0*z4[i] - 26.0*z2[i]*r2[i] + r4[i]) + 1.38512556004858*(x2[i] - y2[i])*(221.0*z6[i] - 195.0*z4[i]*r2[i] + 39.0*z2[i]*r4[i] - r6[i]);
        prCofDX[totalAN*90+i] = 16.3107969549167*x[i]*z[i]*((x2[i] - 3.0*y2[i])*(17.0*z4[i] - 10.0*z2[i]*r2[i] + r4[i]) - (-x2[i] - y2[i] + 4.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]));
        prCofDX[totalAN*91+i] = -1.94951311615607*x2[i]*(-x2[i] - y2[i] + 14.0*z2[i])*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]) + 2.43689139519509*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(85.0*z4[i] - 30.0*z2[i]*r2[i] + r4[i]);
        prCofDX[totalAN*92+i] = -15.1008636641708*x[i]*z[i]*(2.0*x6[i] - 21.0*x4[i]*y2[i] - 7.0*x4[i]*z2[i] + 70.0*x2[i]*y2[i]*z2[i] + 7.0*y6[i] - 35.0*y4[i]*z2[i]);
        prCofDX[totalAN*93+i] = -1.08981096268811*x2[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]) - 3.81433836940837*(x2[i] + y2[i] - 16.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
        prCofDX[totalAN*94+i] = 25.4185411916376*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
        prCofDX[totalAN*95+i] = 6.74010856667869*x8[i] - 188.723039867003*x6[i]*y2[i] + 471.807599667509*x4[i]*y4[i] - 188.723039867003*x2[i]*y6[i] + 6.74010856667869*y8[i];

        prCofDY[totalAN*77+i] = 6.74010856667869*x8[i] - 188.723039867003*x6[i]*y2[i] + 471.807599667509*x4[i]*y4[i] - 188.723039867003*x2[i]*y6[i] + 6.74010856667869*y8[i];
        prCofDY[totalAN*78+i] = 25.4185411916376*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
        prCofDY[totalAN*79+i] = -1.08981096268811*y2[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]) - 3.81433836940837*(x2[i] + y2[i] - 16.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
        prCofDY[totalAN*80+i] = 7.5504318320854*x[i]*z[i]*(y2[i]*(-6.0*x4[i] + 20.0*x2[i]*y2[i] - 6.0*y4[i]) + (-3.0*x2[i] - 3.0*y2[i] + 14.0*z2[i])*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]));
        prCofDY[totalAN*81+i] = -1.94951311615607*y2[i]*(-x2[i] - y2[i] + 14.0*z2[i])*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]) + 2.43689139519509*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(85.0*z4[i] - 30.0*z2[i]*r2[i] + r4[i]);
        prCofDY[totalAN*82+i] = 16.3107969549167*x[i]*z[i]*(x6[i] + 3.0*x4[i]*y2[i] - 8.0*x4[i]*z2[i] - 5.0*x2[i]*y4[i] + 8.0*x2[i]*z4[i] - 7.0*y6[i] + 40.0*y4[i]*z2[i] - 24.0*y2[i]*z4[i]);
        prCofDY[totalAN*83+i] = -2.77025112009717*y2[i]*(3.0*x2[i] - y2[i])*(65.0*z4[i] - 26.0*z2[i]*r2[i] + r4[i]) + 1.38512556004858*(x2[i] - y2[i])*(221.0*z6[i] - 195.0*z4[i]*r2[i] + 39.0*z2[i]*r4[i] - r6[i]);
        prCofDY[totalAN*84+i] = -1.209036709703*x[i]*z[i]*(7.0*x6[i] + 63.0*x4[i]*y2[i] - 70.0*x4[i]*z2[i] + 105.0*x2[i]*y4[i] - 420.0*x2[i]*y2[i]*z2[i] + 112.0*x2[i]*z4[i] + 49.0*y6[i] - 350.0*y4[i]*z2[i] + 336.0*y2[i]*z4[i] - 32.0*z6[i]);
        prCofDY[totalAN*85+i] = -3.60874489652473*y2[i]*(143.0*z6[i] - 143.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i]) + 156.658193633065*z8[i] - 258.025260101518*z6[i]*r2[i] + 129.012630050759*z4[i]*r4[i] - 19.848096930886*z2[i]*r6[i] + 0.451093112065591*r8[i];
        prCofDY[totalAN*86+i] = 0.691662763035835*y[i]*z[i]*(-715.0*z6[i] + 1001.0*z4[i]*r2[i] - 385.0*z2[i]*r4[i] + 35.0*r6[i]);
        prCofDY[totalAN*87+i] = -3.60874489652473*x[i]*y[i]*(143.0*z6[i] - 143.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i]);
        prCofDY[totalAN*88+i] = -2.41807341940599*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*z2[i] - 21.0*x2[i]*y4[i] + 70.0*x2[i]*y2[i]*z2[i] - 14.0*y6[i] + 105.0*y4[i]*z2[i] - 112.0*y2[i]*z4[i] + 16.0*z6[i]);
        prCofDY[totalAN*89+i] = 11.0810044803887*x[i]*y[i]*(x4[i]*y2[i] - 3.0*x4[i]*z2[i] + 2.0*x2[i]*y4[i] - 30.0*x2[i]*y2[i]*z2[i] + 20.0*x2[i]*z4[i] + y6[i] - 27.0*y4[i]*z2[i] + 60.0*y2[i]*z4[i] - 16.0*z6[i]);
        prCofDY[totalAN*90+i] = -16.3107969549167*y[i]*z[i]*((3.0*x2[i] - y2[i])*(17.0*z4[i] - 10.0*z2[i]*r2[i] + r4[i]) - (x2[i] + y2[i] - 4.0*z2[i])*(x4[i] - 6.0*x2[i]*y2[i] + y4[i]));
        prCofDY[totalAN*91+i] = -3.89902623231215*x[i]*y[i]*(2.0*x6[i] + 7.0*x4[i]*y2[i] - 63.0*x4[i]*z2[i] - 70.0*x2[i]*y2[i]*z2[i] + 140.0*x2[i]*z4[i] - 5.0*y6[i] + 105.0*y4[i]*z2[i] - 140.0*y2[i]*z4[i]);
        prCofDY[totalAN*92+i] = 15.1008636641708*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*z2[i] - 21.0*x2[i]*y4[i] + 70.0*x2[i]*y2[i]*z2[i] + 2.0*y6[i] - 7.0*y4[i]*z2[i]);
        prCofDY[totalAN*93+i] = 4.35924385075243*x[i]*y[i]*(5.0*x6[i] - 7.0*x4[i]*y2[i] - 84.0*x4[i]*z2[i] - 21.0*x2[i]*y4[i] + 280.0*x2[i]*y2[i]*z2[i] + 7.0*y6[i] - 84.0*y4[i]*z2[i]);
        prCofDY[totalAN*94+i] = -25.4185411916376*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
        prCofDY[totalAN*95+i] = 53.9208685334296*x[i]*y[i]*(-x6[i] + 7.0*x4[i]*y2[i] - 7.0*x2[i]*y4[i] + y6[i]);

        prCofDZ[totalAN*77+i] = 0;
        prCofDZ[totalAN*78+i] = 25.4185411916376*x[i]*y[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
        prCofDZ[totalAN*79+i] = 17.4369754030097*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
        prCofDZ[totalAN*80+i] = -7.5504318320854*x[i]*y[i]*(x2[i] + y2[i] - 14.0*z2[i])*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
        prCofDZ[totalAN*81+i] = 27.293183626185*y[i]*z[i]*(-x2[i] - y2[i] + 4.0*z2[i])*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]);
        prCofDZ[totalAN*82+i] = 16.3107969549167*x[i]*y[i]*(x2[i] - y2[i])*(65.0*z4[i] - 26.0*z2[i]*r2[i] + r4[i]);
        prCofDZ[totalAN*83+i] = 11.0810044803887*y[i]*z[i]*(3.0*x2[i] - y2[i])*(39.0*z4[i] - 26.0*z2[i]*r2[i] + 3.0*r4[i]);
        prCofDZ[totalAN*84+i] = 8.46325696792098*x[i]*y[i]*(143.0*z6[i] - 143.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i]);
        prCofDZ[totalAN*85+i] = 1.03106997043564*y[i]*z[i]*(715.0*z6[i] - 1001.0*z4[i]*r2[i] + 385.0*z2[i]*r4[i] - 35.0*r6[i]);
        prCofDZ[totalAN*86+i] = 556.356235016949*z8[i] - 1038.53163869831*z6[i]*r2[i] + 599.152868479792*z4[i]*r4[i] - 108.936885178144*z2[i]*r6[i] + 3.02602458828178*r8[i];
        prCofDZ[totalAN*87+i] = 1.03106997043564*x[i]*z[i]*(715.0*z6[i] - 1001.0*z4[i]*r2[i] + 385.0*z2[i]*r4[i] - 35.0*r6[i]);
        prCofDZ[totalAN*88+i] = 4.23162848396049*(x2[i] - y2[i])*(143.0*z6[i] - 143.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i]);
        prCofDZ[totalAN*89+i] = 11.0810044803887*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(39.0*z4[i] - 26.0*z2[i]*r2[i] + 3.0*r4[i]);
        prCofDZ[totalAN*90+i] = 4.07769923872917*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(65.0*z4[i] - 26.0*z2[i]*r2[i] + r4[i]);
        prCofDZ[totalAN*91+i] = 27.293183626185*x[i]*z[i]*(-x2[i] - y2[i] + 4.0*z2[i])*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]);
        prCofDZ[totalAN*92+i] = -3.7752159160427*(x2[i] + y2[i] - 14.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
        prCofDZ[totalAN*93+i] = 17.4369754030097*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
        prCofDZ[totalAN*94+i] = 3.1773176489547*x8[i] - 88.9648941707315*x6[i]*y2[i] + 222.412235426829*x4[i]*y4[i] - 88.9648941707315*x2[i]*y6[i] + 3.1773176489547*y8[i];
        prCofDZ[totalAN*95+i] = 0;
        }
    if (lMax > 9){ 
        preCoef[totalAN*96+i] = 1.53479023644398*x[i]*y[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
        preCoef[totalAN*97+i] = 3.43189529989171*y[i]*z[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
        preCoef[totalAN*98+i] = -4.45381546176335*x[i]*y[i]*(x2[i] + y2[i] - 18.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
        preCoef[totalAN*99+i] = 1.36369691122981*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 16.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
        preCoef[totalAN*100+i] = 0.330745082725238*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i]);
        preCoef[totalAN*101+i] = 0.295827395278969*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(323.0*z4[i] - 170.0*z2[i]*r2[i] + 15.0*r4[i]);
        preCoef[totalAN*102+i] = 1.87097672671297*x[i]*y[i]*(x2[i] - y2[i])*(323.0*z6[i] - 255.0*z4[i]*r2[i] + 45.0*z2[i]*r4[i] - r6[i]);
        preCoef[totalAN*103+i] = 0.661490165450475*y[i]*z[i]*(3.0*x2[i] - y2[i])*(323.0*z6[i] - 357.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 7.0*r6[i]);
        preCoef[totalAN*104+i] = 0.129728894680065*x[i]*y[i]*(4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]);
        preCoef[totalAN*105+i] = 0.0748990122652082*y[i]*z[i]*(4199.0*z8[i] - 7956.0*z6[i]*r2[i] + 4914.0*z4[i]*r4[i] - 1092.0*z2[i]*r6[i] + 63.0*r8[i]);
        preCoef[totalAN*106+i] = 233.240148813258*z10[i] - 552.410878768242*z8[i]*r2[i] + 454.926606044435*z6[i]*r4[i] - 151.642202014812*z4[i]*r6[i] + 17.4971771555552*z2[i]*r8[i] - 0.318130493737367*r10[i];
        preCoef[totalAN*107+i] = 0.0748990122652082*x[i]*z[i]*(4199.0*z8[i] - 7956.0*z6[i]*r2[i] + 4914.0*z4[i]*r4[i] - 1092.0*z2[i]*r6[i] + 63.0*r8[i]);
        preCoef[totalAN*108+i] = 0.0648644473400325*(x2[i] - y2[i])*(4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]);
        preCoef[totalAN*109+i] = 0.661490165450475*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(323.0*z6[i] - 357.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 7.0*r6[i]);
        preCoef[totalAN*110+i] = 0.467744181678242*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(323.0*z6[i] - 255.0*z4[i]*r2[i] + 45.0*z2[i]*r4[i] - r6[i]);
        preCoef[totalAN*111+i] = 0.295827395278969*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(323.0*z4[i] - 170.0*z2[i]*r2[i] + 15.0*r4[i]);
        preCoef[totalAN*112+i] = 0.165372541362619*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
        preCoef[totalAN*113+i] = 1.36369691122981*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 16.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
        preCoef[totalAN*114+i] = -0.556726932720418*(x2[i] + y2[i] - 18.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
        preCoef[totalAN*115+i] = 3.43189529989171*x[i]*z[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
        preCoef[totalAN*116+i] = 0.76739511822199*x10[i] - 34.5327803199895*x8[i]*y2[i] + 161.152974826618*x6[i]*y4[i] - 161.152974826618*x4[i]*y6[i] + 34.5327803199895*x2[i]*y8[i] - 0.76739511822199*y10[i];

        if(return_derivatives){
        prCofDX[totalAN*96+i] = 7.6739511822199*y[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
        prCofDX[totalAN*97+i] = 247.096461592203*x[i]*y[i]*z[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
        prCofDX[totalAN*98+i] = -4.45381546176335*y[i]*(2.0*x2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (x2[i] + y2[i] - 18.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]));
        prCofDX[totalAN*99+i] = 10.9095752898384*x[i]*y[i]*z[i]*(-21.0*x6[i] + 63.0*x4[i]*y2[i] + 84.0*x4[i]*z2[i] + 21.0*x2[i]*y4[i] - 280.0*x2[i]*y2[i]*z2[i] - 15.0*y6[i] + 84.0*y4[i]*z2[i]); //..//
        prCofDX[totalAN*100+i] = 0.992235248175713*y[i]*(-4.0*x2[i]*(-x2[i] - y2[i] + 16.0*z2[i])*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]) + (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i]));
        prCofDX[totalAN*101+i] = 11.8330958111588*x[i]*y[i]*z[i]*(15.0*x6[i] - 105.0*x4[i]*z2[i] - 21.0*x2[i]*y4[i] + 70.0*x2[i]*y2[i]*z2[i] + 84.0*x2[i]*z4[i] - 6.0*y6[i] + 63.0*y4[i]*z2[i] - 84.0*y2[i]*z4[i]);
        prCofDX[totalAN*102+i] = -1.87097672671297*y[i]*(9.0*x8[i] + 14.0*x6[i]*y2[i] - 294.0*x6[i]*z2[i] - 210.0*x4[i]*y2[i]*z2[i] + 840.0*x4[i]*z4[i] - 6.0*x2[i]*y6[i] + 126.0*x2[i]*y4[i]*z2[i] - 336.0*x2[i]*z6[i] - y8[i] + 42.0*y6[i]*z2[i] - 168.0*y4[i]*z4[i] + 112.0*y2[i]*z6[i]);
        prCofDX[totalAN*103+i] = -15.8757639708114*x[i]*y[i]*z[i]*(7.0*x6[i] + 14.0*x4[i]*y2[i] - 63.0*x4[i]*z2[i] + 7.0*x2[i]*y4[i] - 70.0*x2[i]*y2[i]*z2[i] + 84.0*x2[i]*z4[i] - 7.0*y4[i]*z2[i] + 28.0*y2[i]*z4[i] - 16.0*z6[i]);
        prCofDX[totalAN*104+i] = 0.129728894680065*y[i]*(-56.0*x2[i]*(221.0*z6[i] - 195.0*z4[i]*r2[i] + 39.0*z2[i]*r4[i] - r6[i]) + 4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]);
        prCofDX[totalAN*105+i] = -5.39272888309499*x[i]*y[i]*z[i]*(221.0*z6[i] - 273.0*z4[i]*r2[i] + 91.0*z2[i]*r4[i] - 7.0*r6[i]);
        prCofDX[totalAN*106+i] = 0.454472133910524*x[i]*(-2431.0*z8[i] + 4004.0*z6[i]*r2[i] - 2002.0*z4[i]*r4[i] + 308.0*z2[i]*r6[i] - 7.0*r8[i]);
        prCofDX[totalAN*107+i] = 0.0748990122652082*z[i]*(-72.0*x2[i]*(221.0*z6[i] - 273.0*z4[i]*r2[i] + 91.0*z2[i]*r4[i] - 7.0*r6[i]) + 4199.0*z8[i] - 7956.0*z6[i]*r2[i] + 4914.0*z4[i]*r4[i] - 1092.0*z2[i]*r6[i] + 63.0*r8[i]);
        prCofDX[totalAN*108+i] = 0.129728894680065*x[i]*(35.0*x8[i] + 84.0*x6[i]*y2[i] - 1344.0*x6[i]*z2[i] + 42.0*x4[i]*y4[i] - 2016.0*x4[i]*y2[i]*z2[i] + 5040.0*x4[i]*z4[i] - 28.0*x2[i]*y6[i] + 3360.0*x2[i]*y2[i]*z4[i] - 3584.0*x2[i]*z6[i] - 21.0*y8[i] + 672.0*y6[i]*z2[i] - 1680.0*y4[i]*z4[i] + 384.0*z8[i]);
        prCofDX[totalAN*109+i] = 1.98447049635143*z[i]*(-14.0*x2[i]*(x2[i] - 3.0*y2[i])*(17.0*z4[i] - 10.0*z2[i]*r2[i] + r4[i]) + (x2[i] - y2[i])*(323.0*z6[i] - 357.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 7.0*r6[i]));
        prCofDX[totalAN*110+i] = 0.935488363356484*x[i]*(2.0*(x2[i] - 3.0*y2[i])*(323.0*z6[i] - 255.0*z4[i]*r2[i] + 45.0*z2[i]*r4[i] - r6[i]) - 3.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(85.0*z4[i] - 30.0*z2[i]*r2[i] + r4[i]));
        prCofDX[totalAN*111+i] = 1.47913697639485*z[i]*(-4.0*x2[i]*(-3.0*x2[i] - 3.0*y2[i] + 14.0*z2[i])*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i]) + (x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(323.0*z4[i] - 170.0*z2[i]*r2[i] + 15.0*r4[i]));
        prCofDX[totalAN*112+i] = 0.992235248175713*x[i]*(-2.0*(-x2[i] - y2[i] + 16.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]) + (x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i]));
        prCofDX[totalAN*113+i] = 1.36369691122981*z[i]*(-6.0*x2[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]) + 7.0*(-3.0*x2[i] - 3.0*y2[i] + 16.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]));
        prCofDX[totalAN*114+i] = 1.11345386544084*x[i]*(-x8[i] + 28.0*x6[i]*y2[i] - 70.0*x4[i]*y4[i] + 28.0*x2[i]*y6[i] - y8[i] - 4.0*(x2[i] + y2[i] - 18.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]));
        prCofDX[totalAN*115+i] = 30.8870576990254*z[i]*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
        prCofDX[totalAN*116+i] = 7.6739511822199*x[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);

        prCofDY[totalAN*96+i] = 7.6739511822199*x[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
        prCofDY[totalAN*97+i] = 30.8870576990254*z[i]*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
        prCofDY[totalAN*98+i] = -4.45381546176335*x[i]*(2.0*y2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (x2[i] + y2[i] - 18.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]));
        prCofDY[totalAN*99+i] = 1.36369691122981*z[i]*(-6.0*y2[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]) + 7.0*(-3.0*x2[i] - 3.0*y2[i] + 16.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]));
        prCofDY[totalAN*100+i] = 0.992235248175713*x[i]*(-4.0*y2[i]*(-x2[i] - y2[i] + 16.0*z2[i])*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]) + (x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i]));
        prCofDY[totalAN*101+i] = 1.47913697639485*z[i]*(-4.0*y2[i]*(-3.0*x2[i] - 3.0*y2[i] + 14.0*z2[i])*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i]) + (x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(323.0*z4[i] - 170.0*z2[i]*r2[i] + 15.0*r4[i]));
        prCofDY[totalAN*102+i] = -1.87097672671297*x[i]*(x8[i] + 6.0*x6[i]*y2[i] - 42.0*x6[i]*z2[i] - 126.0*x4[i]*y2[i]*z2[i] + 168.0*x4[i]*z4[i] - 14.0*x2[i]*y6[i] + 210.0*x2[i]*y4[i]*z2[i] - 112.0*x2[i]*z6[i] - 9.0*y8[i] + 294.0*y6[i]*z2[i] - 840.0*y4[i]*z4[i] + 336.0*y2[i]*z6[i]);
        prCofDY[totalAN*103+i] = 1.98447049635143*z[i]*(-14.0*y2[i]*(3.0*x2[i] - y2[i])*(17.0*z4[i] - 10.0*z2[i]*r2[i] + r4[i]) + (x2[i] - y2[i])*(323.0*z6[i] - 357.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 7.0*r6[i]));
        prCofDY[totalAN*104+i] = 0.129728894680065*x[i]*(-56.0*y2[i]*(221.0*z6[i] - 195.0*z4[i]*r2[i] + 39.0*z2[i]*r4[i] - r6[i]) + 4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]);
        prCofDY[totalAN*105+i] = 0.0748990122652082*z[i]*(-72.0*y2[i]*(221.0*z6[i] - 273.0*z4[i]*r2[i] + 91.0*z2[i]*r4[i] - 7.0*r6[i]) + 4199.0*z8[i] - 7956.0*z6[i]*r2[i] + 4914.0*z4[i]*r4[i] - 1092.0*z2[i]*r6[i] + 63.0*r8[i]);
        prCofDY[totalAN*106+i] = 0.454472133910524*y[i]*(-2431.0*z8[i] + 4004.0*z6[i]*r2[i] - 2002.0*z4[i]*r4[i] + 308.0*z2[i]*r6[i] - 7.0*r8[i]);
        prCofDY[totalAN*107+i] = -5.39272888309499*x[i]*y[i]*z[i]*(221.0*z6[i] - 273.0*z4[i]*r2[i] + 91.0*z2[i]*r4[i] - 7.0*r6[i]);
        prCofDY[totalAN*108+i] = 0.129728894680065*y[i]*(21.0*x8[i] + 28.0*x6[i]*y2[i] - 672.0*x6[i]*z2[i] - 42.0*x4[i]*y4[i] + 1680.0*x4[i]*z4[i] - 84.0*x2[i]*y6[i] + 2016.0*x2[i]*y4[i]*z2[i] - 3360.0*x2[i]*y2[i]*z4[i] - 35.0*y8[i] + 1344.0*y6[i]*z2[i] - 5040.0*y4[i]*z4[i] + 3584.0*y2[i]*z6[i] - 384.0*z8[i]);
        prCofDY[totalAN*109+i] = 15.8757639708114*x[i]*y[i]*z[i]*(7.0*x4[i]*y2[i] - 7.0*x4[i]*z2[i] + 14.0*x2[i]*y4[i] - 70.0*x2[i]*y2[i]*z2[i] + 28.0*x2[i]*z4[i] + 7.0*y6[i] - 63.0*y4[i]*z2[i] + 84.0*y2[i]*z4[i] - 16.0*z6[i]);
        prCofDY[totalAN*110+i] = -0.935488363356484*y[i]*(2.0*(3.0*x2[i] - y2[i])*(323.0*z6[i] - 255.0*z4[i]*r2[i] + 45.0*z2[i]*r4[i] - r6[i]) + 3.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(85.0*z4[i] - 30.0*z2[i]*r2[i] + r4[i]));
        prCofDY[totalAN*111+i] = -11.8330958111588*x[i]*y[i]*z[i]*(6.0*x6[i] + 21.0*x4[i]*y2[i] - 63.0*x4[i]*z2[i] - 70.0*x2[i]*y2[i]*z2[i] + 84.0*x2[i]*z4[i] - 15.0*y6[i] + 105.0*y4[i]*z2[i] - 84.0*y2[i]*z4[i]);
        prCofDY[totalAN*112+i] = 0.992235248175713*y[i]*(2.0*(x2[i] + y2[i] - 16.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]) - (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i]));
        prCofDY[totalAN*113+i] = 10.9095752898384*x[i]*y[i]*z[i]*(15.0*x6[i] - 21.0*x4[i]*y2[i] - 84.0*x4[i]*z2[i] - 63.0*x2[i]*y4[i] + 280.0*x2[i]*y2[i]*z2[i] + 21.0*y6[i] - 84.0*y4[i]*z2[i]);
        prCofDY[totalAN*114+i] = 1.11345386544084*y[i]*(-x8[i] + 28.0*x6[i]*y2[i] - 70.0*x4[i]*y4[i] + 28.0*x2[i]*y6[i] - y8[i] + 4.0*(x2[i] + y2[i] - 18.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]));
        prCofDY[totalAN*115+i] = -247.096461592203*x[i]*y[i]*z[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
        prCofDY[totalAN*116+i] = 7.6739511822199*y[i]*(-9.0*x8[i] + 84.0*x6[i]*y2[i] - 126.0*x4[i]*y4[i] + 36.0*x2[i]*y6[i] - y8[i]);

        prCofDZ[totalAN*96+i] = 0;
        prCofDZ[totalAN*97+i] = 3.43189529989171*y[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
        prCofDZ[totalAN*98+i] = 160.337356623481*x[i]*y[i]*z[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
        prCofDZ[totalAN*99+i] = -4.09109073368942*y[i]*(x2[i] + y2[i] - 16.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
        prCofDZ[totalAN*100+i] = 21.1676852944152*x[i]*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 14.0*z2[i])*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]);
        prCofDZ[totalAN*101+i] = 4.43741092918453*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(85.0*z4[i] - 30.0*z2[i]*r2[i] + r4[i]);
        prCofDZ[totalAN*102+i] = 157.162045043889*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(17.0*z4[i] - 10.0*z2[i]*r2[i] + r4[i]);
        prCofDZ[totalAN*103+i] = 4.63043115815333*y[i]*(3.0*x2[i] - y2[i])*(221.0*z6[i] - 195.0*z4[i]*r2[i] + 39.0*z2[i]*r4[i] - r6[i]);
        prCofDZ[totalAN*104+i] = 12.4539738892862*x[i]*y[i]*z[i]*(221.0*z6[i] - 273.0*z4[i]*r2[i] + 91.0*z2[i]*r4[i] - 7.0*r6[i]);
        prCofDZ[totalAN*105+i] = 0.674091110386874*y[i]*(2431.0*z8[i] - 4004.0*z6[i]*r2[i] + 2002.0*z4[i]*r4[i] - 308.0*z2[i]*r6[i] + 7.0*r8[i]);
        prCofDZ[totalAN*106+i] = 0.100993807535672*z[i]*(12155.0*z8[i] - 25740.0*z6[i]*r2[i] + 18018.0*z4[i]*r4[i] - 4620.0*z2[i]*r6[i] + 315.0*r8[i]);
        prCofDZ[totalAN*107+i] = 0.674091110386874*x[i]*(2431.0*z8[i] - 4004.0*z6[i]*r2[i] + 2002.0*z4[i]*r4[i] - 308.0*z2[i]*r6[i] + 7.0*r8[i]);
        prCofDZ[totalAN*108+i] = 6.22698694464312*z[i]*(x2[i] - y2[i])*(221.0*z6[i] - 273.0*z4[i]*r2[i] + 91.0*z2[i]*r4[i] - 7.0*r6[i]);
        prCofDZ[totalAN*109+i] = 4.63043115815333*x[i]*(x2[i] - 3.0*y2[i])*(221.0*z6[i] - 195.0*z4[i]*r2[i] + 39.0*z2[i]*r4[i] - r6[i]);
        prCofDZ[totalAN*110+i] = 39.2905112609723*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(17.0*z4[i] - 10.0*z2[i]*r2[i] + r4[i]);
        prCofDZ[totalAN*111+i] = 4.43741092918453*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(85.0*z4[i] - 30.0*z2[i]*r2[i] + r4[i]);
        prCofDZ[totalAN*112+i] = 10.5838426472076*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 14.0*z2[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
        prCofDZ[totalAN*113+i] = -4.09109073368942*x[i]*(x2[i] + y2[i] - 16.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
        prCofDZ[totalAN*114+i] = 20.0421695779351*z[i]*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
        prCofDZ[totalAN*115+i] = 3.43189529989171*x[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
        prCofDZ[totalAN*116+i] = 0;
        }
    if (lMax > 10){ 
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

        if(return_derivatives){
      prCofDX[totalAN*117+i] = 17.2621263273183*x[i]*y[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      prCofDX[totalAN*118+i] = 36.8029769880531*y[i]*z[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      prCofDX[totalAN*119+i] = 1.13576452756687*x[i]*y[i]*(-9.0*x8[i] + 84.0*x6[i]*y2[i] - 126.0*x4[i]*y4[i] + 36.0*x2[i]*y6[i] - y8[i] - 36.0*(x2[i] + y2[i] - 20.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])) ;
      prCofDX[totalAN*120+i] = 35.1903768038371*y[i]*z[i]*(-2.0*x2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (-x2[i] - y2[i] + 6.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])) ;
      prCofDX[totalAN*121+i] = 1.00915326495424*x[i]*y[i]*(-2.0*(-x2[i] - y2[i] + 18.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]) + 7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i]));
      prCofDX[totalAN*122+i] = 0.638244565090152*y[i]*z[i]*(-20.0*x2[i]*(-3.0*x2[i] - 3.0*y2[i] + 16.0*z2[i])*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]) + 3.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(399.0*z4[i] - 190.0*z2[i]*r2[i] + 15.0*r4[i]));
      prCofDX[totalAN*123+i] = 0.947934431913346*x[i]*y[i]*(2.0*(x2[i] - y2[i])*(2261.0*z6[i] - 1615.0*z4[i]*r2[i] + 255.0*z2[i]*r4[i] - 5.0*r6[i]) - (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i]));
      prCofDX[totalAN*124+i] = -4.0127980256608*y[i]*z[i]*(45.0*x8[i] + 70.0*x6[i]*y2[i] - 490.0*x6[i]*z2[i] - 350.0*x4[i]*y2[i]*z2[i] + 840.0*x4[i]*z4[i] - 30.0*x2[i]*y6[i] + 210.0*x2[i]*y4[i]*z2[i] - 240.0*x2[i]*z6[i] - 5.0*y8[i] + 70.0*y6[i]*z2[i] - 168.0*y4[i]*z4[i] + 80.0*y2[i]*z6[i]);
      prCofDX[totalAN*125+i] = 0.915791665569424*x[i]*y[i]*(15.0*x8[i] + 44.0*x6[i]*y2[i] - 672.0*x6[i]*z2[i] + 42.0*x4[i]*y4[i] - 1344.0*x4[i]*y2[i]*z2[i] + 3024.0*x4[i]*z4[i] + 12.0*x2[i]*y6[i] - 672.0*x2[i]*y4[i]*z2[i] + 3360.0*x2[i]*y2[i]*z4[i] - 2688.0*x2[i]*z6[i] - y8[i] + 336.0*y4[i]*z4[i] - 896.0*y2[i]*z6[i] + 384.0*z8[i]);
      prCofDX[totalAN*126+i] = 0.489511235746264*y[i]*z[i]*(-24.0*x2[i]*(323.0*z6[i] - 357.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 7.0*r6[i]) + 2261.0*z8[i] - 3876.0*z6[i]*r2[i] + 2142.0*z4[i]*r4[i] - 420.0*z2[i]*r6[i] + 21.0*r8[i]);
      prCofDX[totalAN*127+i] = -0.643994632278231*x[i]*y[i]*(4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]);
      prCofDX[totalAN*128+i] = 0.581315236111974*x[i]*z[i]*(-4199.0*z8[i] + 7956.0*z6[i]*r2[i] - 4914.0*z4[i]*r4[i] + 1092.0*z2[i]*r6[i] - 63.0*r8[i]);
      prCofDX[totalAN*129+i] = -0.643994632278231*x2[i]*(4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]) + 630.964474218468*z10[i] - 1352.06673046815*z8[i]*r2[i] + 996.259696134424*z6[i]*r4[i] - 293.017557686595*z4[i]*r6[i] + 29.3017557686595*z2[i]*r8[i] - 0.450796242594762*r10[i];
      prCofDX[totalAN*130+i] = 0.489511235746264*x[i]*z[i]*(105.0*x8[i] + 252.0*x6[i]*y2[i] - 1344.0*x6[i]*z2[i] + 126.0*x4[i]*y4[i] - 2016.0*x4[i]*y2[i]*z2[i] + 3024.0*x4[i]*z4[i] - 84.0*x2[i]*y6[i] + 2016.0*x2[i]*y2[i]*z4[i] - 1536.0*x2[i]*z6[i] - 63.0*y8[i] + 672.0*y6[i]*z2[i] - 1008.0*y4[i]*z4[i] + 128.0*z8[i]);
      prCofDX[totalAN*131+i] = -3.66316666227769*x2[i]*(x2[i] - 3.0*y2[i])*(323.0*z6[i] - 255.0*z4[i]*r2[i] + 45.0*z2[i]*r4[i] - r6[i]) + 1.37368749835414*(x2[i] - y2[i])*(969.0*z8[i] - 1292.0*z6[i]*r2[i] + 510.0*z4[i]*r4[i] - 60.0*z2[i]*r6[i] + r8[i]) ;
      prCofDX[totalAN*132+i] = 2.0063990128304*x[i]*z[i]*(2.0*(x2[i] - 3.0*y2[i])*(323.0*z6[i] - 323.0*z4[i]*r2[i] + 85.0*z2[i]*r4[i] - 5.0*r6[i]) - (x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(323.0*z4[i] - 170.0*z2[i]*r2[i] + 15.0*r4[i]));
      prCofDX[totalAN*133+i] = -0.947934431913346*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i]) + 0.473967215956673*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(2261.0*z6[i] - 1615.0*z4[i]*r2[i] + 255.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDX[totalAN*134+i] = 0.638244565090152*x[i]*z[i]*(75.0*x8[i] - 780.0*x6[i]*y2[i] - 640.0*x6[i]*z2[i] - 630.0*x4[i]*y4[i] + 6720.0*x4[i]*y2[i]*z2[i] + 672.0*x4[i]*z4[i] + 420.0*x2[i]*y6[i] - 6720.0*x2[i]*y2[i]*z4[i] + 195.0*y8[i] - 2240.0*y6[i]*z2[i] + 3360.0*y4[i]*z4[i]);
      prCofDX[totalAN*135+i] = -2.01830652990847*x2[i]*(-x2[i] - y2[i] + 18.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]) + 3.53203642733983*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
      prCofDX[totalAN*136+i] = 8.79759420095928*x[i]*z[i]*(-x8[i] + 28.0*x6[i]*y2[i] - 70.0*x4[i]*y4[i] + 28.0*x2[i]*y6[i] - y8[i] + 4.0*(-x2[i] - y2[i] + 6.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]));
      prCofDX[totalAN*137+i] = -1.13576452756687*x2[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) - 5.11094037405094*(x2[i] + y2[i] - 20.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDX[totalAN*138+i] = 36.8029769880531*x[i]*z[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      prCofDX[totalAN*139+i] = 8.63106316365917*x10[i] - 388.397842364662*x8[i]*y2[i] + 1812.52326436842*x6[i]*y4[i] - 1812.52326436842*x4[i]*y6[i] + 388.397842364662*x2[i]*y8[i] - 8.63106316365917*y10[i];

      prCofDY[totalAN*117+i] = 8.63106316365917*x10[i] - 388.397842364662*x8[i]*y2[i] + 1812.52326436842*x6[i]*y4[i] - 1812.52326436842*x4[i]*y6[i] + 388.397842364662*x2[i]*y8[i] - 8.63106316365917*y10[i];
      prCofDY[totalAN*118+i] = 36.8029769880531*x[i]*z[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      prCofDY[totalAN*119+i] = -1.13576452756687*y2[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) - 5.11094037405094*(x2[i] + y2[i] - 20.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDY[totalAN*120+i] = 35.1903768038371*x[i]*z[i]*(-2.0*y2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (-x2[i] - y2[i] + 6.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]));
      prCofDY[totalAN*121+i] = -2.01830652990847*y2[i]*(-x2[i] - y2[i] + 18.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]) + 3.53203642733983*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
      prCofDY[totalAN*122+i] = 0.638244565090152*x[i]*z[i]*(-20.0*y2[i]*(-3.0*x2[i] - 3.0*y2[i] + 16.0*z2[i])*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i]) + 3.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(399.0*z4[i] - 190.0*z2[i]*r2[i] + 15.0*r4[i]));
      prCofDY[totalAN*123+i] = -0.947934431913346*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i]) + 0.473967215956673*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(2261.0*z6[i] - 1615.0*z4[i]*r2[i] + 255.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDY[totalAN*124+i] = -4.0127980256608*x[i]*z[i]*(5.0*x8[i] + 30.0*x6[i]*y2[i] - 70.0*x6[i]*z2[i] - 210.0*x4[i]*y2[i]*z2[i] + 168.0*x4[i]*z4[i] - 70.0*x2[i]*y6[i] + 350.0*x2[i]*y4[i]*z2[i] - 80.0*x2[i]*z6[i] - 45.0*y8[i] + 490.0*y6[i]*z2[i] - 840.0*y4[i]*z4[i] + 240.0*y2[i]*z6[i]);
      prCofDY[totalAN*125+i] = -3.66316666227769*y2[i]*(3.0*x2[i] - y2[i])*(323.0*z6[i] - 255.0*z4[i]*r2[i] + 45.0*z2[i]*r4[i] - r6[i]) + 1.37368749835414*(x2[i] - y2[i])*(969.0*z8[i] - 1292.0*z6[i]*r2[i] + 510.0*z4[i]*r4[i] - 60.0*z2[i]*r6[i] + r8[i]);
      prCofDY[totalAN*126+i] = 0.489511235746264*x[i]*z[i]*(-24.0*y2[i]*(323.0*z6[i] - 357.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 7.0*r6[i]) + 2261.0*z8[i] - 3876.0*z6[i]*r2[i] + 2142.0*z4[i]*r4[i] - 420.0*z2[i]*r6[i] + 21.0*r8[i]);
      prCofDY[totalAN*127+i] = -0.643994632278231*y2[i]*(4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]) + 630.964474218468*z10[i] - 1352.06673046815*z8[i]*r2[i] + 996.259696134424*z6[i]*r4[i] - 293.017557686595*z4[i]*r6[i] + 29.3017557686595*z2[i]*r8[i] - 0.450796242594762*r10[i];
      prCofDY[totalAN*128+i] = 0.581315236111974*y[i]*z[i]*(-4199.0*z8[i] + 7956.0*z6[i]*r2[i] - 4914.0*z4[i]*r4[i] + 1092.0*z2[i]*r6[i] - 63.0*r8[i]);
      prCofDY[totalAN*129+i] = -0.643994632278231*x[i]*y[i]*(4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]);
      prCofDY[totalAN*130+i] = 0.489511235746264*y[i]*z[i]*(63.0*x8[i] + 84.0*x6[i]*y2[i] - 672.0*x6[i]*z2[i] - 126.0*x4[i]*y4[i] + 1008.0*x4[i]*z4[i] - 252.0*x2[i]*y6[i] + 2016.0*x2[i]*y4[i]*z2[i] - 2016.0*x2[i]*y2[i]*z4[i] - 105.0*y8[i] + 1344.0*y6[i]*z2[i] - 3024.0*y4[i]*z4[i] + 1536.0*y2[i]*z6[i] - 128.0*z8[i]);
      prCofDY[totalAN*131+i] = 0.915791665569424*x[i]*y[i]*(x8[i] - 12.0*x6[i]*y2[i] - 42.0*x4[i]*y4[i] + 672.0*x4[i]*y2[i]*z2[i] - 336.0*x4[i]*z4[i] - 44.0*x2[i]*y6[i] + 1344.0*x2[i]*y4[i]*z2[i] - 3360.0*x2[i]*y2[i]*z4[i] + 896.0*x2[i]*z6[i] - 15.0*y8[i] + 672.0*y6[i]*z2[i] - 3024.0*y4[i]*z4[i] + 2688.0*y2[i]*z6[i] - 384.0*z8[i]);
      prCofDY[totalAN*132+i] = -2.0063990128304*y[i]*z[i]*(2.0*(3.0*x2[i] - y2[i])*(323.0*z6[i] - 323.0*z4[i]*r2[i] + 85.0*z2[i]*r4[i] - 5.0*r6[i]) + (x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(323.0*z4[i] - 170.0*z2[i]*r2[i] + 15.0*r4[i]));
      prCofDY[totalAN*133+i] = 0.947934431913346*x[i]*y[i]*(7.0*x8[i] + 44.0*x6[i]*y2[i] - 384.0*x6[i]*z2[i] + 42.0*x4[i]*y4[i] - 1344.0*x4[i]*y2[i]*z2[i] + 2016.0*x4[i]*z4[i] - 20.0*x2[i]*y6[i] + 2240.0*x2[i]*y2[i]*z4[i] - 1792.0*x2[i]*z6[i] - 25.0*y8[i] + 960.0*y6[i]*z2[i] - 3360.0*y4[i]*z4[i] + 1792.0*y2[i]*z6[i]);
      prCofDY[totalAN*134+i] = -0.638244565090152*y[i]*z[i]*(195.0*x8[i] + 420.0*x6[i]*y2[i] - 2240.0*x6[i]*z2[i] - 630.0*x4[i]*y4[i] + 3360.0*x4[i]*z4[i] - 780.0*x2[i]*y6[i] + 6720.0*x2[i]*y4[i]*z2[i] - 6720.0*x2[i]*y2[i]*z4[i] + 75.0*y8[i] - 640.0*y6[i]*z2[i] + 672.0*y4[i]*z4[i]);
      prCofDY[totalAN*135+i] = 1.00915326495424*x[i]*y[i]*(2.0*(x2[i] + y2[i] - 18.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]) - 7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i]));
      prCofDY[totalAN*136+i] = 8.79759420095928*y[i]*z[i]*(-x8[i] + 28.0*x6[i]*y2[i] - 70.0*x4[i]*y4[i] + 28.0*x2[i]*y6[i] - y8[i] - 4.0*(-x2[i] - y2[i] + 6.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]));
      prCofDY[totalAN*137+i] = 1.13576452756687*x[i]*y[i]*(-x8[i] + 36.0*x6[i]*y2[i] - 126.0*x4[i]*y4[i] + 84.0*x2[i]*y6[i] - 9.0*y8[i] + 36.0*(x2[i] + y2[i] - 20.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]));
      prCofDY[totalAN*138+i] = -36.8029769880531*y[i]*z[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      prCofDY[totalAN*139+i] = 17.2621263273183*x[i]*y[i]*(-5.0*x8[i] + 60.0*x6[i]*y2[i] - 126.0*x4[i]*y4[i] + 60.0*x2[i]*y6[i] - 5.0*y8[i]);

      prCofDZ[totalAN*117+i] = 0;
      prCofDZ[totalAN*118+i] = 7.36059539761062*x[i]*y[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      prCofDZ[totalAN*119+i] = 22.7152905513375*y[i]*z[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*120+i] = -35.1903768038371*x[i]*y[i]*(x2[i] + y2[i] - 18.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
      prCofDZ[totalAN*121+i] = 12.1098391794508*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 16.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
      prCofDZ[totalAN*122+i] = 3.19122282545076*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i]);
      prCofDZ[totalAN*123+i] = 3.03339018212271*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(323.0*z4[i] - 170.0*z2[i]*r2[i] + 15.0*r4[i]);
      prCofDZ[totalAN*124+i] = 20.063990128304*x[i]*y[i]*(x2[i] - y2[i])*(323.0*z6[i] - 255.0*z4[i]*r2[i] + 45.0*z2[i]*r4[i] - r6[i]);
      prCofDZ[totalAN*125+i] = 7.32633332455539*y[i]*z[i]*(3.0*x2[i] - y2[i])*(323.0*z6[i] - 357.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 7.0*r6[i]);
      prCofDZ[totalAN*126+i] = 1.46853370723879*x[i]*y[i]*(4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]);
      prCofDZ[totalAN*127+i] = 0.858659509704308*y[i]*z[i]*(4199.0*z8[i] - 7956.0*z6[i]*r2[i] + 4914.0*z4[i]*r4[i] - 1092.0*z2[i]*r6[i] + 63.0*r8[i]);
      prCofDZ[totalAN*128+i] = 2685.03694407759*z10[i] - 6359.29802544694*z8[i]*r2[i] + 5237.06896213277*z6[i]*r4[i] - 1745.68965404426*z4[i]*r6[i] + 201.425729312799*z2[i]*r8[i] - 3.66228598750543*r10[i];
      prCofDZ[totalAN*129+i] = 0.858659509704308*x[i]*z[i]*(4199.0*z8[i] - 7956.0*z6[i]*r2[i] + 4914.0*z4[i]*r4[i] - 1092.0*z2[i]*r6[i] + 63.0*r8[i]);
      prCofDZ[totalAN*130+i] = 0.734266853619395*(x2[i] - y2[i])*(4199.0*z8[i] - 6188.0*z6[i]*r2[i] + 2730.0*z4[i]*r4[i] - 364.0*z2[i]*r6[i] + 7.0*r8[i]);
      prCofDZ[totalAN*131+i] = 7.32633332455539*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(323.0*z6[i] - 357.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 7.0*r6[i]);
      prCofDZ[totalAN*132+i] = 5.01599753207601*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(323.0*z6[i] - 255.0*z4[i]*r2[i] + 45.0*z2[i]*r4[i] - r6[i]);
      prCofDZ[totalAN*133+i] = 3.03339018212271*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(323.0*z4[i] - 170.0*z2[i]*r2[i] + 15.0*r4[i]);
      prCofDZ[totalAN*134+i] = 1.59561141272538*(323.0*z4[i] - 102.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
      prCofDZ[totalAN*135+i] = 12.1098391794508*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 16.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
      prCofDZ[totalAN*136+i] = -4.39879710047964*(x2[i] + y2[i] - 18.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*137+i] = 22.7152905513375*x[i]*z[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      prCofDZ[totalAN*138+i] = 3.68029769880531*x10[i] - 165.613396446239*x8[i]*y2[i] + 772.862516749115*x6[i]*y4[i] - 772.862516749115*x4[i]*y6[i] + 165.613396446239*x2[i]*y8[i] - 3.68029769880531*y10[i];
      prCofDZ[totalAN*139+i] = 0;
        }
    if (lMax > 11){ 
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

        if(return_derivatives){
      prCofDX[totalAN*140+i] = 9.60986394940766*y[i]*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      prCofDX[totalAN*141+i] = 86.3106316365917*x[i]*y[i]*z[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      prCofDX[totalAN*142+i] = 1.15689166958762*y[i]*(x2[i]*(-10.0*x8[i] + 120.0*x6[i]*y2[i] - 252.0*x4[i]*y4[i] + 120.0*x2[i]*y6[i] - 10.0*y8[i]) - 5.0*(x2[i] + y2[i] - 22.0*z2[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*143+i] = 9.39863235373324*x[i]*y[i]*z[i]*(-9.0*x8[i] + 84.0*x6[i]*y2[i] - 126.0*x4[i]*y4[i] + 36.0*x2[i]*y6[i] - y8[i] + 12.0*(-3.0*x2[i] - 3.0*y2[i] + 20.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]));
      prCofDX[totalAN*144+i] = 4.10189944667082*y[i]*(-4.0*x2[i]*(-x2[i] - y2[i] + 20.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]));
      prCofDX[totalAN*145+i] = 2.05094972333541*x[i]*y[i]*z[i]*(-10.0*(-x2[i] - y2[i] + 6.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]) + 7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i]));
      prCofDX[totalAN*146+i] = 0.576267123343003*y[i]*(-10.0*x2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i]) + (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDX[totalAN*147+i] = 2.15619413881132*x[i]*y[i]*z[i]*(10.0*(x2[i] - y2[i])*(437.0*z6[i] - 399.0*z4[i]*r2[i] + 95.0*z2[i]*r4[i] - 5.0*r6[i]) - (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(399.0*z4[i] - 190.0*z2[i]*r2[i] + 15.0*r4[i]));
      prCofDX[totalAN*148+i] = 0.369784244098711*y[i]*(-8.0*x2[i]*(x2[i] - y2[i])*(2261.0*z6[i] - 1615.0*z4[i]*r2[i] + 255.0*z2[i]*r4[i] - 5.0*r6[i]) + (3.0*x2[i] - y2[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDX[totalAN*149+i] = 0.739568488197423*x[i]*y[i]*z[i]*(225.0*x8[i] + 660.0*x6[i]*y2[i] - 3360.0*x6[i]*z2[i] + 630.0*x4[i]*y4[i] - 6720.0*x4[i]*y2[i]*z2[i] + 9072.0*x4[i]*z4[i] + 180.0*x2[i]*y6[i] - 3360.0*x2[i]*y4[i]*z2[i] + 10080.0*x2[i]*y2[i]*z4[i] - 5760.0*x2[i]*z6[i] - 15.0*y8[i] + 1008.0*y4[i]*z4[i] - 1920.0*y2[i]*z6[i] + 640.0*z8[i]);
      prCofDX[totalAN*150+i] = 0.301927570987541*y[i]*(-30.0*x2[i]*(969.0*z8[i] - 1292.0*z6[i]*r2[i] + 510.0*z4[i]*r4[i] - 60.0*z2[i]*r6[i] + r8[i]) + 7429.0*z10[i] - 14535.0*z8[i]*r2[i] + 9690.0*z6[i]*r4[i] - 2550.0*z4[i]*r6[i] + 225.0*z2[i]*r8[i] - 3.0*r10[i]);
      prCofDX[totalAN*151+i] = -2.6763018719158*x[i]*y[i]*z[i]*(2261.0*z8[i] - 3876.0*z6[i]*r2[i] + 2142.0*z4[i]*r4[i] - 420.0*z2[i]*r6[i] + 21.0*r8[i]);
      prCofDX[totalAN*152+i] = 0.181818908760507*x[i]*(-29393.0*z10[i] + 62985.0*z8[i]*r2[i] - 46410.0*z6[i]*r4[i] + 13650.0*z4[i]*r6[i] - 1365.0*z2[i]*r8[i] + 21.0*r10[i]);
      prCofDX[totalAN*153+i] = 0.0243300170174164*z[i]*(-110.0*x2[i]*(2261.0*z8[i] - 3876.0*z6[i]*r2[i] + 2142.0*z4[i]*r4[i] - 420.0*z2[i]*r6[i] + 21.0*r8[i]) + 52003.0*z10[i] - 124355.0*z8[i]*r2[i] + 106590.0*z6[i]*r4[i] - 39270.0*z4[i]*r6[i] + 5775.0*z2[i]*r8[i] - 231.0*r10[i]);
      prCofDX[totalAN*154+i] = 0.301927570987541*x[i]*(7429.0*z10[i] - 14535.0*z8[i]*r2[i] + 9690.0*z6[i]*r4[i] - 2550.0*z4[i]*r6[i] + 225.0*z2[i]*r8[i] - 15.0*(x2[i] - y2[i])*(969.0*z8[i] - 1292.0*z6[i]*r2[i] + 510.0*z4[i]*r4[i] - 60.0*z2[i]*r6[i] + r8[i]) - 3.0*r10[i]);
      prCofDX[totalAN*155+i] = 0.369784244098711*z[i]*(-24.0*x2[i]*(x2[i] - 3.0*y2[i])*(323.0*z6[i] - 323.0*z4[i]*r2[i] + 85.0*z2[i]*r4[i] - 5.0*r6[i]) + (x2[i] - y2[i])*(7429.0*z8[i] - 11628.0*z6[i]*r2[i] + 5814.0*z4[i]*r4[i] - 1020.0*z2[i]*r6[i] + 45.0*r8[i]));
      prCofDX[totalAN*156+i] = 0.369784244098711*x[i]*((x2[i] - 3.0*y2[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]) - 2.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(2261.0*z6[i] - 1615.0*z4[i]*r2[i] + 255.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDX[totalAN*157+i] = 1.07809706940566*z[i]*(-2.0*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(399.0*z4[i] - 190.0*z2[i]*r2[i] + 15.0*r4[i]) + 5.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(437.0*z6[i] - 399.0*z4[i]*r2[i] + 95.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDX[totalAN*158+i] = 0.576267123343003*x[i]*((x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]) - 5.0*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]));
      prCofDX[totalAN*159+i] = 1.0254748616677*z[i]*(-20.0*x2[i]*(-x2[i] - y2[i] + 6.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]) + 7.0*(161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]));
      prCofDX[totalAN*160+i] = 2.05094972333541*x[i]*(-(-x2[i] - y2[i] + 20.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]) + 2.0*(161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]));
      prCofDX[totalAN*161+i] = 4.69931617686662*z[i]*(-2.0*x2[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) + 3.0*(-3.0*x2[i] - 3.0*y2[i] + 20.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*162+i] = 1.15689166958762*x[i]*(-x10[i] + 45.0*x8[i]*y2[i] - 210.0*x6[i]*y4[i] + 210.0*x4[i]*y6[i] - 45.0*x2[i]*y8[i] + y10[i] - 5.0*(x2[i] + y2[i] - 22.0*z2[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDX[totalAN*163+i] = 43.1553158182958*z[i]*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDX[totalAN*164+i] = 9.60986394940766*x[i]*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);

      prCofDY[totalAN*140+i] = 9.60986394940766*x[i]*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      prCofDY[totalAN*141+i] = 43.1553158182958*z[i]*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDY[totalAN*142+i] = 1.15689166958762*x[i]*(y2[i]*(-10.0*x8[i] + 120.0*x6[i]*y2[i] - 252.0*x4[i]*y4[i] + 120.0*x2[i]*y6[i] - 10.0*y8[i]) - 5.0*(x2[i] + y2[i] - 22.0*z2[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDY[totalAN*143+i] = 4.69931617686662*z[i]*(-2.0*y2[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) + 3.0*(-3.0*x2[i] - 3.0*y2[i] + 20.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]));
      prCofDY[totalAN*144+i] = 4.10189944667082*x[i]*(-4.0*y2[i]*(-x2[i] - y2[i] + 20.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]));
      prCofDY[totalAN*145+i] = 1.0254748616677*z[i]*(-20.0*y2[i]*(-x2[i] - y2[i] + 6.0*z2[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]) + 7.0*(161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]));
      prCofDY[totalAN*146+i] = 0.576267123343003*x[i]*(-10.0*y2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i]) + (x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDY[totalAN*147+i] = 1.07809706940566*z[i]*(-2.0*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(399.0*z4[i] - 190.0*z2[i]*r2[i] + 15.0*r4[i]) + 5.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(437.0*z6[i] - 399.0*z4[i]*r2[i] + 95.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDY[totalAN*148+i] = 0.369784244098711*x[i]*(-8.0*y2[i]*(x2[i] - y2[i])*(2261.0*z6[i] - 1615.0*z4[i]*r2[i] + 255.0*z2[i]*r4[i] - 5.0*r6[i]) + (x2[i] - 3.0*y2[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDY[totalAN*149+i] = 0.369784244098711*z[i]*(-24.0*y2[i]*(3.0*x2[i] - y2[i])*(323.0*z6[i] - 323.0*z4[i]*r2[i] + 85.0*z2[i]*r4[i] - 5.0*r6[i]) + (x2[i] - y2[i])*(7429.0*z8[i] - 11628.0*z6[i]*r2[i] + 5814.0*z4[i]*r4[i] - 1020.0*z2[i]*r6[i] + 45.0*r8[i]));
      prCofDY[totalAN*150+i] = 0.301927570987541*x[i]*(-30.0*y2[i]*(969.0*z8[i] - 1292.0*z6[i]*r2[i] + 510.0*z4[i]*r4[i] - 60.0*z2[i]*r6[i] + r8[i]) + 7429.0*z10[i] - 14535.0*z8[i]*r2[i] + 9690.0*z6[i]*r4[i] - 2550.0*z4[i]*r6[i] + 225.0*z2[i]*r8[i] - 3.0*r10[i]);
      prCofDY[totalAN*151+i] = 0.0243300170174164*z[i]*(-110.0*y2[i]*(2261.0*z8[i] - 3876.0*z6[i]*r2[i] + 2142.0*z4[i]*r4[i] - 420.0*z2[i]*r6[i] + 21.0*r8[i]) + 52003.0*z10[i] - 124355.0*z8[i]*r2[i] + 106590.0*z6[i]*r4[i] - 39270.0*z4[i]*r6[i] + 5775.0*z2[i]*r8[i] - 231.0*r10[i]);
      prCofDY[totalAN*152+i] = 0.181818908760507*y[i]*(-29393.0*z10[i] + 62985.0*z8[i]*r2[i] - 46410.0*z6[i]*r4[i] + 13650.0*z4[i]*r6[i] - 1365.0*z2[i]*r8[i] + 21.0*r10[i]);
      prCofDY[totalAN*153+i] = -2.6763018719158*x[i]*y[i]*z[i]*(2261.0*z8[i] - 3876.0*z6[i]*r2[i] + 2142.0*z4[i]*r4[i] - 420.0*z2[i]*r6[i] + 21.0*r8[i]);
      prCofDY[totalAN*154+i] = 0.301927570987541*y[i]*(-7429.0*z10[i] + 14535.0*z8[i]*r2[i] - 9690.0*z6[i]*r4[i] + 2550.0*z4[i]*r6[i] - 225.0*z2[i]*r8[i] - 15.0*(x2[i] - y2[i])*(969.0*z8[i] - 1292.0*z6[i]*r2[i] + 510.0*z4[i]*r4[i] - 60.0*z2[i]*r6[i] + r8[i]) + 3.0*r10[i]);
      prCofDY[totalAN*155+i] = 0.739568488197423*x[i]*y[i]*z[i]*(15.0*x8[i] - 180.0*x6[i]*y2[i] - 630.0*x4[i]*y4[i] + 3360.0*x4[i]*y2[i]*z2[i] - 1008.0*x4[i]*z4[i] - 660.0*x2[i]*y6[i] + 6720.0*x2[i]*y4[i]*z2[i] - 10080.0*x2[i]*y2[i]*z4[i] + 1920.0*x2[i]*z6[i] - 225.0*y8[i] + 3360.0*y6[i]*z2[i] - 9072.0*y4[i]*z4[i] + 5760.0*y2[i]*z6[i] - 640.0*z8[i]);
      prCofDY[totalAN*156+i] = -0.369784244098711*y[i]*((3.0*x2[i] - y2[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]) + 2.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(2261.0*z6[i] - 1615.0*z4[i]*r2[i] + 255.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDY[totalAN*157+i] = 2.15619413881132*x[i]*y[i]*z[i]*(35.0*x8[i] + 220.0*x6[i]*y2[i] - 640.0*x6[i]*z2[i] + 210.0*x4[i]*y4[i] - 2240.0*x4[i]*y2[i]*z2[i] + 2016.0*x4[i]*z4[i] - 100.0*x2[i]*y6[i] + 2240.0*x2[i]*y2[i]*z4[i] - 1280.0*x2[i]*z6[i] - 125.0*y8[i] + 1600.0*y6[i]*z2[i] - 3360.0*y4[i]*z4[i] + 1280.0*y2[i]*z6[i]);
      prCofDY[totalAN*158+i] = -0.576267123343003*y[i]*((5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]) + 5.0*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]));
      prCofDY[totalAN*159+i] = 2.05094972333541*x[i]*y[i]*z[i]*(10.0*(x2[i] + y2[i] - 6.0*z2[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]) - 7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i]));
      prCofDY[totalAN*160+i] = 2.05094972333541*y[i]*((x2[i] + y2[i] - 20.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]) - 2.0*(161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]));
      prCofDY[totalAN*161+i] = 9.39863235373324*x[i]*y[i]*z[i]*(-x8[i] + 36.0*x6[i]*y2[i] - 126.0*x4[i]*y4[i] + 84.0*x2[i]*y6[i] - 9.0*y8[i] - 12.0*(-3.0*x2[i] - 3.0*y2[i] + 20.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]));
      prCofDY[totalAN*162+i] = 1.15689166958762*y[i]*(-x10[i] + 45.0*x8[i]*y2[i] - 210.0*x6[i]*y4[i] + 210.0*x4[i]*y6[i] - 45.0*x2[i]*y8[i] + y10[i] + 5.0*(x2[i] + y2[i] - 22.0*z2[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDY[totalAN*163+i] = -86.3106316365917*x[i]*y[i]*z[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      prCofDY[totalAN*164+i] = 9.60986394940766*y[i]*(-11.0*x10[i] + 165.0*x8[i]*y2[i] - 462.0*x6[i]*y4[i] + 330.0*x4[i]*y6[i] - 55.0*x2[i]*y8[i] + y10[i]);

      prCofDZ[totalAN*140+i] = 0;
      prCofDZ[totalAN*141+i] = 3.92321052893598*y[i]*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*142+i] = 50.9032334618553*x[i]*y[i]*z[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      prCofDZ[totalAN*143+i] = -4.69931617686662*y[i]*(x2[i] + y2[i] - 20.0*z2[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*144+i] = 328.151955733665*x[i]*y[i]*z[i]*(-x2[i] - y2[i] + 6.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
      prCofDZ[totalAN*145+i] = 5.12737430833852*y[i]*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
      prCofDZ[totalAN*146+i] = 6.91520548011604*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(399.0*z4[i] - 190.0*z2[i]*r2[i] + 15.0*r4[i]);
      prCofDZ[totalAN*147+i] = 1.07809706940566*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(2261.0*z6[i] - 1615.0*z4[i]*r2[i] + 255.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDZ[totalAN*148+i] = 47.332383244635*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(323.0*z6[i] - 323.0*z4[i]*r2[i] + 85.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDZ[totalAN*149+i] = 5.54676366148067*y[i]*(3.0*x2[i] - y2[i])*(969.0*z8[i] - 1292.0*z6[i]*r2[i] + 510.0*z4[i]*r4[i] - 60.0*z2[i]*r6[i] + r8[i]);
      prCofDZ[totalAN*150+i] = 6.03855141975083*x[i]*y[i]*z[i]*(2261.0*z8[i] - 3876.0*z6[i]*r2[i] + 2142.0*z4[i]*r4[i] - 420.0*z2[i]*r6[i] + 21.0*r8[i]);
      prCofDZ[totalAN*151+i] = 0.26763018719158*y[i]*(29393.0*z10[i] - 62985.0*z8[i]*r2[i] + 46410.0*z6[i]*r4[i] - 13650.0*z4[i]*r6[i] + 1365.0*z2[i]*r8[i] - 21.0*r10[i]);
      prCofDZ[totalAN*152+i] = 0.0661159668220027*z[i]*(88179.0*z10[i] - 230945.0*z8[i]*r2[i] + 218790.0*z6[i]*r4[i] - 90090.0*z4[i]*r6[i] + 15015.0*z2[i]*r8[i] - 693.0*r10[i]);
      prCofDZ[totalAN*153+i] = 0.26763018719158*x[i]*(29393.0*z10[i] - 62985.0*z8[i]*r2[i] + 46410.0*z6[i]*r4[i] - 13650.0*z4[i]*r6[i] + 1365.0*z2[i]*r8[i] - 21.0*r10[i]);
      prCofDZ[totalAN*154+i] = 3.01927570987541*z[i]*(x2[i] - y2[i])*(2261.0*z8[i] - 3876.0*z6[i]*r2[i] + 2142.0*z4[i]*r4[i] - 420.0*z2[i]*r6[i] + 21.0*r8[i]);
      prCofDZ[totalAN*155+i] = 5.54676366148067*x[i]*(x2[i] - 3.0*y2[i])*(969.0*z8[i] - 1292.0*z6[i]*r2[i] + 510.0*z4[i]*r4[i] - 60.0*z2[i]*r6[i] + r8[i]);
      prCofDZ[totalAN*156+i] = 11.8330958111588*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(323.0*z6[i] - 323.0*z4[i]*r2[i] + 85.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDZ[totalAN*157+i] = 1.07809706940566*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(2261.0*z6[i] - 1615.0*z4[i]*r2[i] + 255.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDZ[totalAN*158+i] = 3.45760274005802*z[i]*(399.0*z4[i] - 190.0*z2[i]*r2[i] + 15.0*r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]);
      prCofDZ[totalAN*159+i] = 5.12737430833852*x[i]*(133.0*z4[i] - 38.0*z2[i]*r2[i] + r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
      prCofDZ[totalAN*160+i] = 41.0189944667082*z[i]*(-x2[i] - y2[i] + 6.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*161+i] = -4.69931617686662*x[i]*(x2[i] + y2[i] - 20.0*z2[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      prCofDZ[totalAN*162+i] = 25.4516167309276*z[i]*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*163+i] = 3.92321052893598*x[i]*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      prCofDZ[totalAN*164+i] = 0;
        }
    if (lMax > 12){ 
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

        if(return_derivatives){
      prCofDX[totalAN*165+i] = 42.4360101795567*x[i]*y[i]*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      prCofDX[totalAN*166+i] = 49.9343178425957*y[i]*z[i]*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      prCofDX[totalAN*167+i] = 1.1769631586808*x[i]*y[i]*(-11.0*x10[i] + 165.0*x8[i]*y2[i] - 462.0*x6[i]*y4[i] + 330.0*x4[i]*y6[i] - 55.0*x2[i]*y8[i] + y10[i] - 11.0*(x2[i] + y2[i] - 24.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDX[totalAN*168+i] = 3.32895452283972*y[i]*z[i]*(x2[i]*(-30.0*x8[i] + 360.0*x6[i]*y2[i] - 756.0*x4[i]*y4[i] + 360.0*x2[i]*y6[i] - 30.0*y8[i]) + 5.0*(-3.0*x2[i] - 3.0*y2[i] + 22.0*z2[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*169+i] = 2.08240500525772*x[i]*y[i]*(-(-x2[i] - y2[i] + 22.0*z2[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) + 6.0*(575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]));
      prCofDX[totalAN*170+i] = 14.560298633254*y[i]*z[i]*(-4.0*x2[i]*(-3.0*x2[i] - 3.0*y2[i] + 20.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]));
      prCofDX[totalAN*171+i] = 0.972850873834812*x[i]*y[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]) - 3.0*(161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]));
      prCofDX[totalAN*172+i] = 6.90655606399791*y[i]*z[i]*(-2.0*x2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i]) + (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDX[totalAN*173+i] = 0.373463779940342*x[i]*y[i]*(5.0*(x2[i] - y2[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i]) - 2.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDX[totalAN*174+i] = 0.528157542646753*y[i]*z[i]*(-72.0*x2[i]*(x2[i] - y2[i])*(437.0*z6[i] - 399.0*z4[i]*r2[i] + 95.0*z2[i]*r4[i] - 5.0*r6[i]) + (3.0*x2[i] - y2[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]));
      prCofDX[totalAN*175+i] = 0.303808757854291*x[i]*y[i]*(37145.0*z10[i] - 66861.0*z8[i]*r2[i] + 40698.0*z6[i]*r4[i] - 9690.0*z4[i]*r6[i] + 765.0*z2[i]*r8[i] - 3.0*(3.0*x2[i] - y2[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]) - 9.0*r10[i]);
      prCofDX[totalAN*176+i] = 0.122135716100197*y[i]*z[i]*(-22.0*x2[i]*(7429.0*z8[i] - 11628.0*z6[i]*r2[i] + 5814.0*z4[i]*r4[i] - 1020.0*z2[i]*r6[i] + 45.0*r8[i]) + 37145.0*z10[i] - 81719.0*z8[i]*r2[i] + 63954.0*z6[i]*r4[i] - 21318.0*z4[i]*r6[i] + 2805.0*z2[i]*r8[i] - 99.0*r10[i]);
      prCofDX[totalAN*177+i] = -1.80248484029233*x[i]*y[i]*(7429.0*z10[i] - 14535.0*z8[i]*r2[i] + 9690.0*z6[i]*r4[i] - 2550.0*z4[i]*r6[i] + 225.0*z2[i]*r8[i] - 3.0*r10[i]);
      prCofDX[totalAN*178+i] = 0.223306616768131*x[i]*z[i]*(-52003.0*z10[i] + 124355.0*z8[i]*r2[i] - 106590.0*z6[i]*r4[i] + 39270.0*z4[i]*r6[i] - 5775.0*z2[i]*r8[i] + 231.0*r10[i]);
      prCofDX[totalAN*179+i] = -1.80248484029233*x2[i]*(7429.0*z10[i] - 14535.0*z8[i]*r2[i] + 9690.0*z6[i]*r4[i] - 2550.0*z4[i]*r6[i] + 225.0*z2[i]*r8[i] - 3.0*r10[i]) + 2536.10982547949*z12[i] - 6695.32993926584*z10[i]*r2[i] + 6549.77928841224*z8[i]*r4[i] - 2911.0130170721*z6[i]*r6[i] + 574.542042843179*z4[i]*r8[i] - 40.5559089065773*z2[i]*r10[i] + 0.450621210073081*r12[i];
      prCofDX[totalAN*180+i] = -0.244271432200393*x[i]*z[i]*(297.0*x10[i] + 990.0*x8[i]*y2[i] - 5775.0*x8[i]*z2[i] + 990.0*x6[i]*y4[i] - 13860.0*x6[i]*y2[i]*z2[i] + 22176.0*x6[i]*z4[i] - 6930.0*x4[i]*y4[i]*z2[i] + 33264.0*x4[i]*y2[i]*z4[i] - 23760.0*x4[i]*z6[i] - 495.0*x2[i]*y8[i] + 4620.0*x2[i]*y6[i]*z2[i] - 15840.0*x2[i]*y2[i]*z6[i] + 7040.0*x2[i]*z8[i] - 198.0*y10[i] + 3465.0*y8[i]*z2[i] - 11088.0*y6[i]*z4[i] + 7920.0*y4[i]*z6[i] - 384.0*z10[i]);
      prCofDX[totalAN*181+i] = -0.911426273562874*x2[i]*(x2[i] - 3.0*y2[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]) + 0.151904378927146*(x2[i] - y2[i])*(37145.0*z10[i] - 66861.0*z8[i]*r2[i] + 40698.0*z6[i]*r4[i] - 9690.0*z4[i]*r6[i] + 765.0*z2[i]*r8[i] - 9.0*r10[i]);
      prCofDX[totalAN*182+i] = 0.528157542646753*x[i]*z[i]*((x2[i] - 3.0*y2[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]) - 18.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(437.0*z6[i] - 399.0*z4[i]*r2[i] + 95.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDX[totalAN*183+i] = -0.746927559880684*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]) + 0.466829724925428*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i]);
      prCofDX[totalAN*184+i] = 6.90655606399791*x[i]*z[i]*((x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]) - (161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]));
      prCofDX[totalAN*185+i] = -2.91855262150443*x2[i]*(161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]) + 3.40497805842184*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]);
      prCofDX[totalAN*186+i] = 7.28014931662701*x[i]*z[i]*(-(-3.0*x2[i] - 3.0*y2[i] + 20.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]) + 2.0*(115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]));
      prCofDX[totalAN*187+i] = -2.08240500525772*x2[i]*(-x2[i] - y2[i] + 22.0*z2[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) + 1.56180375394329*(575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDX[totalAN*188+i] = -6.65790904567943*x[i]*z[i]*(9.0*x10[i] - 330.0*x8[i]*y2[i] - 55.0*x8[i]*z2[i] + 990.0*x6[i]*y4[i] + 1980.0*x6[i]*y2[i]*z2[i] - 6930.0*x4[i]*y4[i]*z2[i] - 495.0*x2[i]*y8[i] + 4620.0*x2[i]*y6[i]*z2[i] + 66.0*y10[i] - 495.0*y8[i]*z2[i]);
      prCofDX[totalAN*189+i] = -1.1769631586808*x2[i]*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) - 6.47329737274437*(x2[i] + y2[i] - 24.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDX[totalAN*190+i] = 49.9343178425957*x[i]*z[i]*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      prCofDX[totalAN*191+i] = 10.6090025448892*x12[i] - 700.194167962685*x10[i]*y2[i] + 5251.45625972014*x8[i]*y4[i] - 9802.71835147759*x6[i]*y6[i] + 5251.45625972014*x4[i]*y8[i] - 700.194167962685*x2[i]*y10[i] + 10.6090025448892*y12[i];

      prCofDY[totalAN*165+i] = 10.6090025448892*x12[i] - 700.194167962685*x10[i]*y2[i] + 5251.45625972014*x8[i]*y4[i] - 9802.71835147759*x6[i]*y6[i] + 5251.45625972014*x4[i]*y8[i] - 700.194167962685*x2[i]*y10[i] + 10.6090025448892*y12[i];
      prCofDY[totalAN*166+i] = 49.9343178425957*x[i]*z[i]*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      prCofDY[totalAN*167+i] = -1.1769631586808*y2[i]*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) - 6.47329737274437*(x2[i] + y2[i] - 24.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDY[totalAN*168+i] = 3.32895452283972*x[i]*z[i]*(y2[i]*(-30.0*x8[i] + 360.0*x6[i]*y2[i] - 756.0*x4[i]*y4[i] + 360.0*x2[i]*y6[i] - 30.0*y8[i]) + 5.0*(-3.0*x2[i] - 3.0*y2[i] + 22.0*z2[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDY[totalAN*169+i] = -2.08240500525772*y2[i]*(-x2[i] - y2[i] + 22.0*z2[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) + 1.56180375394329*(575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDY[totalAN*170+i] = 14.560298633254*x[i]*z[i]*(-4.0*y2[i]*(-3.0*x2[i] - 3.0*y2[i] + 20.0*z2[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]));
      prCofDY[totalAN*171+i] = -2.91855262150443*y2[i]*(161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]) + 3.40497805842184*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]);
      prCofDY[totalAN*172+i] = 6.90655606399791*x[i]*z[i]*(-2.0*y2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i]) + (x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDY[totalAN*173+i] = -0.746927559880684*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]) + 0.466829724925428*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i]);
      prCofDY[totalAN*174+i] = 0.528157542646753*x[i]*z[i]*(-72.0*y2[i]*(x2[i] - y2[i])*(437.0*z6[i] - 399.0*z4[i]*r2[i] + 95.0*z2[i]*r4[i] - 5.0*r6[i]) + (x2[i] - 3.0*y2[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]));
      prCofDY[totalAN*175+i] = -0.911426273562874*y2[i]*(3.0*x2[i] - y2[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]) + 0.151904378927146*(x2[i] - y2[i])*(37145.0*z10[i] - 66861.0*z8[i]*r2[i] + 40698.0*z6[i]*r4[i] - 9690.0*z4[i]*r6[i] + 765.0*z2[i]*r8[i] - 9.0*r10[i]);
      prCofDY[totalAN*176+i] = 0.122135716100197*x[i]*z[i]*(-22.0*y2[i]*(7429.0*z8[i] - 11628.0*z6[i]*r2[i] + 5814.0*z4[i]*r4[i] - 1020.0*z2[i]*r6[i] + 45.0*r8[i]) + 37145.0*z10[i] - 81719.0*z8[i]*r2[i] + 63954.0*z6[i]*r4[i] - 21318.0*z4[i]*r6[i] + 2805.0*z2[i]*r8[i] - 99.0*r10[i]);
      prCofDY[totalAN*177+i] = -1.80248484029233*y2[i]*(7429.0*z10[i] - 14535.0*z8[i]*r2[i] + 9690.0*z6[i]*r4[i] - 2550.0*z4[i]*r6[i] + 225.0*z2[i]*r8[i] - 3.0*r10[i]) + 2536.10982547949*z12[i] - 6695.32993926584*z10[i]*r2[i] + 6549.77928841224*z8[i]*r4[i] - 2911.0130170721*z6[i]*r6[i] + 574.542042843179*z4[i]*r8[i] - 40.5559089065773*z2[i]*r10[i] + 0.450621210073081*r12[i];
      prCofDY[totalAN*178+i] = 0.223306616768131*y[i]*z[i]*(-52003.0*z10[i] + 124355.0*z8[i]*r2[i] - 106590.0*z6[i]*r4[i] + 39270.0*z4[i]*r6[i] - 5775.0*z2[i]*r8[i] + 231.0*r10[i]);
      prCofDY[totalAN*179+i] = -1.80248484029233*x[i]*y[i]*(7429.0*z10[i] - 14535.0*z8[i]*r2[i] + 9690.0*z6[i]*r4[i] - 2550.0*z4[i]*r6[i] + 225.0*z2[i]*r8[i] - 3.0*r10[i]);
      prCofDY[totalAN*180+i] = -0.244271432200393*y[i]*z[i]*(198.0*x10[i] + 495.0*x8[i]*y2[i] - 3465.0*x8[i]*z2[i] - 4620.0*x6[i]*y2[i]*z2[i] + 11088.0*x6[i]*z4[i] - 990.0*x4[i]*y6[i] + 6930.0*x4[i]*y4[i]*z2[i] - 7920.0*x4[i]*z6[i] - 990.0*x2[i]*y8[i] + 13860.0*x2[i]*y6[i]*z2[i] - 33264.0*x2[i]*y4[i]*z4[i] + 15840.0*x2[i]*y2[i]*z6[i] - 297.0*y10[i] + 5775.0*y8[i]*z2[i] - 22176.0*y6[i]*z4[i] + 23760.0*y4[i]*z6[i] - 7040.0*y2[i]*z8[i] + 384.0*z10[i]);
      prCofDY[totalAN*181+i] = 0.303808757854291*x[i]*y[i]*(-37145.0*z10[i] + 66861.0*z8[i]*r2[i] - 40698.0*z6[i]*r4[i] + 9690.0*z4[i]*r6[i] - 765.0*z2[i]*r8[i] - 3.0*(x2[i] - 3.0*y2[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]) + 9.0*r10[i]);
      prCofDY[totalAN*182+i] = -0.528157542646753*y[i]*z[i]*((3.0*x2[i] - y2[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]) + 18.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(437.0*z6[i] - 399.0*z4[i]*r2[i] + 95.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDY[totalAN*183+i] = -0.373463779940342*x[i]*y[i]*(5.0*(x2[i] - y2[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i]) + 2.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDY[totalAN*184+i] = -6.90655606399791*y[i]*z[i]*((5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]) + (161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i])*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i]));
      prCofDY[totalAN*185+i] = -0.972850873834812*x[i]*y[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]) + 3.0*(161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]));
      prCofDY[totalAN*186+i] = 7.28014931662701*y[i]*z[i]*((3.0*x2[i] + 3.0*y2[i] - 20.0*z2[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]) - 2.0*(115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]));
      prCofDY[totalAN*187+i] = 2.08240500525772*x[i]*y[i]*((x2[i] + y2[i] - 22.0*z2[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) - 6.0*(575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]));
      prCofDY[totalAN*188+i] = 6.65790904567943*y[i]*z[i]*(66.0*x10[i] - 495.0*x8[i]*y2[i] - 495.0*x8[i]*z2[i] + 4620.0*x6[i]*y2[i]*z2[i] + 990.0*x4[i]*y6[i] - 6930.0*x4[i]*y4[i]*z2[i] - 330.0*x2[i]*y8[i] + 1980.0*x2[i]*y6[i]*z2[i] + 9.0*y10[i] - 55.0*y8[i]*z2[i]);
      prCofDY[totalAN*189+i] = 1.1769631586808*x[i]*y[i]*(-x10[i] + 55.0*x8[i]*y2[i] - 330.0*x6[i]*y4[i] + 462.0*x4[i]*y6[i] - 165.0*x2[i]*y8[i] + 11.0*y10[i] + 11.0*(x2[i] + y2[i] - 24.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDY[totalAN*190+i] = -49.9343178425957*y[i]*z[i]*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      prCofDY[totalAN*191+i] = 42.4360101795567*x[i]*y[i]*(-3.0*x10[i] + 55.0*x8[i]*y2[i] - 198.0*x6[i]*y4[i] + 198.0*x4[i]*y6[i] - 55.0*x2[i]*y8[i] + 3.0*y10[i]);

      prCofDZ[totalAN*165+i] = 0;
      prCofDZ[totalAN*166+i] = 16.6447726141986*x[i]*y[i]*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      prCofDZ[totalAN*167+i] = 28.2471158083391*y[i]*z[i]*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*168+i] = -9.98686356851915*x[i]*y[i]*(x2[i] + y2[i] - 22.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      prCofDZ[totalAN*169+i] = 15.2709700385566*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 20.0*z2[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*170+i] = 43.6808958997621*x[i]*y[i]*(161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
      prCofDZ[totalAN*171+i] = 11.6742104860177*y[i]*z[i]*(161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]);
      prCofDZ[totalAN*172+i] = 2.30218535466597*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDZ[totalAN*173+i] = 13.4446960778523*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(437.0*z6[i] - 399.0*z4[i]*r2[i] + 95.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDZ[totalAN*174+i] = 4.75341788382078*x[i]*y[i]*(x2[i] - y2[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]);
      prCofDZ[totalAN*175+i] = 1.62031337522289*y[i]*z[i]*(3.0*x2[i] - y2[i])*(7429.0*z8[i] - 11628.0*z6[i]*r2[i] + 5814.0*z4[i]*r4[i] - 1020.0*z2[i]*r6[i] + 45.0*r8[i]);
      prCofDZ[totalAN*176+i] = 4.03047863130649*x[i]*y[i]*(7429.0*z10[i] - 14535.0*z8[i]*r2[i] + 9690.0*z6[i]*r4[i] - 2550.0*z4[i]*r6[i] + 225.0*z2[i]*r8[i] - 3.0*r10[i]);
      prCofDZ[totalAN*177+i] = 0.327724516416786*y[i]*z[i]*(52003.0*z10[i] - 124355.0*z8[i]*r2[i] + 106590.0*z6[i]*r4[i] - 39270.0*z4[i]*r6[i] + 5775.0*z2[i]*r8[i] - 231.0*r10[i]);
      prCofDZ[totalAN*178+i] = 12580.3318244426*z12[i] - 36100.0826266613*z10[i]*r2[i] + 38678.6599571371*z8[i]*r4[i] - 19000.0434877165*z6[i]*r6[i] + 4191.18606346687*z4[i]*r8[i] - 335.294885077349*z2[i]*r10[i] + 4.29865237278653*r12[i];
      prCofDZ[totalAN*179+i] = 0.327724516416786*x[i]*z[i]*(52003.0*z10[i] - 124355.0*z8[i]*r2[i] + 106590.0*z6[i]*r4[i] - 39270.0*z4[i]*r6[i] + 5775.0*z2[i]*r8[i] - 231.0*r10[i]);
      prCofDZ[totalAN*180+i] = 2.01523931565325*(x2[i] - y2[i])*(7429.0*z10[i] - 14535.0*z8[i]*r2[i] + 9690.0*z6[i]*r4[i] - 2550.0*z4[i]*r6[i] + 225.0*z2[i]*r8[i] - 3.0*r10[i]);
      prCofDZ[totalAN*181+i] = 1.62031337522289*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(7429.0*z8[i] - 11628.0*z6[i]*r2[i] + 5814.0*z4[i]*r4[i] - 1020.0*z2[i]*r6[i] + 45.0*r8[i]);
      prCofDZ[totalAN*182+i] = 1.18835447095519*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(7429.0*z8[i] - 9044.0*z6[i]*r2[i] + 3230.0*z4[i]*r4[i] - 340.0*z2[i]*r6[i] + 5.0*r8[i]);
      prCofDZ[totalAN*183+i] = 13.4446960778523*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(437.0*z6[i] - 399.0*z4[i]*r2[i] + 95.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDZ[totalAN*184+i] = 1.15109267733299*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(3059.0*z6[i] - 1995.0*z4[i]*r2[i] + 285.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDZ[totalAN*185+i] = 11.6742104860177*x[i]*z[i]*(161.0*z4[i] - 70.0*z2[i]*r2[i] + 5.0*r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]);
      prCofDZ[totalAN*186+i] = 5.46011198747026*(161.0*z4[i] - 42.0*z2[i]*r2[i] + r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*187+i] = 15.2709700385566*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 20.0*z2[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      prCofDZ[totalAN*188+i] = -4.99343178425957*(x2[i] + y2[i] - 22.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*189+i] = 28.2471158083391*x[i]*z[i]*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      prCofDZ[totalAN*190+i] = 4.16119315354964*x12[i] - 274.638748134277*x10[i]*y2[i] + 2059.79061100707*x8[i]*y4[i] - 3844.94247387987*x6[i]*y6[i] + 2059.79061100707*x4[i]*y8[i] - 274.638748134277*x2[i]*y10[i] + 4.16119315354964*y12[i];
      prCofDZ[totalAN*191+i] = 0;
        }
    if (lMax > 13){ 
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

        if(return_derivatives){
      prCofDX[totalAN*192+i] = 11.6273091629033*y[i]*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      prCofDX[totalAN*193+i] = 228.52490857415*x[i]*y[i]*z[i]*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      prCofDX[totalAN*194+i] = 2.39217700650788*y[i]*(x2[i]*(-6.0*x10[i] + 110.0*x8[i]*y2[i] - 396.0*x6[i]*y4[i] + 396.0*x4[i]*y6[i] - 110.0*x2[i]*y8[i] + 6.0*y10[i]) - 3.0*(x2[i] + y2[i] - 26.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDX[totalAN*195+i] = 10.5635676357028*x[i]*y[i]*z[i]*(-11.0*x10[i] + 165.0*x8[i]*y2[i] - 462.0*x6[i]*y4[i] + 330.0*x4[i]*y6[i] - 55.0*x2[i]*y8[i] + y10[i] + 11.0*(-x2[i] - y2[i] + 8.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDX[totalAN*196+i] = 1.05635676357028*y[i]*(-4.0*x2[i]*(-x2[i] - y2[i] + 24.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + 5.0*(225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*197+i] = 7.71453904240792*x[i]*y[i]*z[i]*(-(-3.0*x2[i] - 3.0*y2[i] + 22.0*z2[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) + 18.0*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]));
      prCofDX[totalAN*198+i] = 3.94023104498585*y[i]*(-2.0*x2[i]*(575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i]));
      prCofDX[totalAN*199+i] = 12.224245339519*x[i]*y[i]*z[i]*((3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]) - (115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]));
      prCofDX[totalAN*200+i] = 0.943121003323134*y[i]*(-8.0*x2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]) + 3.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]));
      prCofDX[totalAN*201+i] = 5.061318418652*x[i]*y[i]*z[i]*(5.0*(x2[i] - y2[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]) - 2.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDX[totalAN*202+i] = 1.83593315348819*y[i]*(-2.0*x2[i]*(x2[i] - y2[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i]) + (3.0*x2[i] - y2[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]));
      prCofDX[totalAN*203+i] = 0.130474087834899*x[i]*y[i]*z[i]*(176985.0*z10[i] - 360525.0*z8[i]*r2[i] + 259578.0*z6[i]*r4[i] - 79002.0*z4[i]*r6[i] + 9405.0*z2[i]*r8[i] - 11.0*(3.0*x2[i] - y2[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]) - 297.0*r10[i]);
      prCofDX[totalAN*204+i] = 0.0274050400015396*y[i]*(-44.0*x2[i]*(37145.0*z10[i] - 66861.0*z8[i]*r2[i] + 40698.0*z6[i]*r4[i] - 9690.0*z4[i]*r6[i] + 765.0*z2[i]*r8[i] - 9.0*r10[i]) + 334305.0*z12[i] - 817190.0*z10[i]*r2[i] + 735471.0*z8[i]*r4[i] - 298452.0*z6[i]*r6[i] + 53295.0*z4[i]*r8[i] - 3366.0*z2[i]*r10[i] + 33.0*r12[i]);
      prCofDX[totalAN*205+i] = -0.790482215453542*x[i]*y[i]*z[i]*(37145.0*z10[i] - 81719.0*z8[i]*r2[i] + 63954.0*z6[i]*r4[i] - 21318.0*z4[i]*r6[i] + 2805.0*z2[i]*r8[i] - 99.0*r10[i]);
      prCofDX[totalAN*206+i] = 0.135000539051985*x[i]*(-185725.0*z12[i] + 490314.0*z10[i]*r2[i] - 479655.0*z8[i]*r4[i] + 213180.0*z6[i]*r6[i] - 42075.0*z4[i]*r8[i] + 2970.0*z2[i]*r10[i] - 33.0*r12[i]);
      prCofDX[totalAN*207+i] = 0.0152015810664143*z[i]*(-52.0*x2[i]*(37145.0*z10[i] - 81719.0*z8[i]*r2[i] + 63954.0*z6[i]*r4[i] - 21318.0*z4[i]*r6[i] + 2805.0*z2[i]*r8[i] - 99.0*r10[i]) + 334305.0*z12[i] - 965770.0*z10[i]*r2[i] + 1062347.0*z8[i]*r4[i] - 554268.0*z6[i]*r6[i] + 138567.0*z4[i]*r8[i] - 14586.0*z2[i]*r10[i] + 429.0*r12[i]);
      prCofDX[totalAN*208+i] = 0.0274050400015396*x[i]*(334305.0*z12[i] - 817190.0*z10[i]*r2[i] + 735471.0*z8[i]*r4[i] - 298452.0*z6[i]*r6[i] + 53295.0*z4[i]*r8[i] - 3366.0*z2[i]*r10[i] - 22.0*(x2[i] - y2[i])*(37145.0*z10[i] - 66861.0*z8[i]*r2[i] + 40698.0*z6[i]*r4[i] - 9690.0*z4[i]*r6[i] + 765.0*z2[i]*r8[i] - 9.0*r10[i]) + 33.0*r12[i]);
      prCofDX[totalAN*209+i] = 0.0652370439174496*z[i]*(-22.0*x2[i]*(x2[i] - 3.0*y2[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]) + 3.0*(x2[i] - y2[i])*(58995.0*z10[i] - 120175.0*z8[i]*r2[i] + 86526.0*z6[i]*r4[i] - 26334.0*z4[i]*r6[i] + 3135.0*z2[i]*r8[i] - 99.0*r10[i]));
      prCofDX[totalAN*210+i] = 0.917966576744097*x[i]*(2.0*(x2[i] - 3.0*y2[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]) - (x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i])) ;
      prCofDX[totalAN*211+i] = 1.265329604663*z[i]*(-8.0*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]) + 5.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDX[totalAN*212+i] = 0.943121003323134*x[i]*(3.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]) - 4.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]));
      prCofDX[totalAN*213+i] = 6.11212266975949*z[i]*(-2.0*x2[i]*(115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]) + (x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]));
      prCofDX[totalAN*214+i] = 0.985057761246463*x[i]*(-(575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]) + 4.0*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i]));
      prCofDX[totalAN*215+i] = 1.92863476060198*z[i]*(-4.0*x2[i]*(-3.0*x2[i] - 3.0*y2[i] + 22.0*z2[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) + 9.0*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*216+i] = 1.05635676357028*x[i]*(-2.0*(-x2[i] - y2[i] + 24.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) + 5.0*(225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDX[totalAN*217+i] = 5.2817838178514*z[i]*(-2.0*x2[i]*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) + 11.0*(-x2[i] - y2[i] + 8.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]));
      prCofDX[totalAN*218+i] = 1.19608850325394*x[i]*(-x12[i] + 66.0*x10[i]*y2[i] - 495.0*x8[i]*y4[i] + 924.0*x6[i]*y6[i] - 495.0*x4[i]*y8[i] + 66.0*x2[i]*y10[i] - y12[i] - 6.0*(x2[i] + y2[i] - 26.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDX[totalAN*219+i] = 57.1312271435375*z[i]*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDX[totalAN*220+i] = 11.6273091629033*x[i]*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);

      prCofDY[totalAN*192+i] = 11.6273091629033*x[i]*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      prCofDY[totalAN*193+i] = 57.1312271435375*z[i]*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDY[totalAN*194+i] = 2.39217700650788*x[i]*(y2[i]*(-6.0*x10[i] + 110.0*x8[i]*y2[i] - 396.0*x6[i]*y4[i] + 396.0*x4[i]*y6[i] - 110.0*x2[i]*y8[i] + 6.0*y10[i]) - 3.0*(x2[i] + y2[i] - 26.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDY[totalAN*195+i] = 5.2817838178514*z[i]*(-2.0*y2[i]*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 11.0*(-x2[i] - y2[i] + 8.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]));
      prCofDY[totalAN*196+i] = 1.05635676357028*x[i]*(-4.0*y2[i]*(-x2[i] - y2[i] + 24.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + 5.0*(225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDY[totalAN*197+i] = 1.92863476060198*z[i]*(-4.0*y2[i]*(-3.0*x2[i] - 3.0*y2[i] + 22.0*z2[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) + 9.0*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]));
      prCofDY[totalAN*198+i] = 3.94023104498585*x[i]*(-2.0*y2[i]*(575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i]));
      prCofDY[totalAN*199+i] = 6.11212266975949*z[i]*(-2.0*y2[i]*(115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i]) + (x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]));
      prCofDY[totalAN*200+i] = 0.943121003323134*x[i]*(-8.0*y2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]) + 3.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]));
      prCofDY[totalAN*201+i] = 1.265329604663*z[i]*(-8.0*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]) + 5.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDY[totalAN*202+i] = 1.83593315348819*x[i]*(-2.0*y2[i]*(x2[i] - y2[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i]) + (x2[i] - 3.0*y2[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]));
      prCofDY[totalAN*203+i] = 0.0652370439174496*z[i]*(-22.0*y2[i]*(3.0*x2[i] - y2[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]) + 3.0*(x2[i] - y2[i])*(58995.0*z10[i] - 120175.0*z8[i]*r2[i] + 86526.0*z6[i]*r4[i] - 26334.0*z4[i]*r6[i] + 3135.0*z2[i]*r8[i] - 99.0*r10[i]));
      prCofDY[totalAN*204+i] = 0.0274050400015396*x[i]*(-44.0*y2[i]*(37145.0*z10[i] - 66861.0*z8[i]*r2[i] + 40698.0*z6[i]*r4[i] - 9690.0*z4[i]*r6[i] + 765.0*z2[i]*r8[i] - 9.0*r10[i]) + 334305.0*z12[i] - 817190.0*z10[i]*r2[i] + 735471.0*z8[i]*r4[i] - 298452.0*z6[i]*r6[i] + 53295.0*z4[i]*r8[i] - 3366.0*z2[i]*r10[i] + 33.0*r12[i]);
      prCofDY[totalAN*205+i] = 0.0152015810664143*z[i]*(-52.0*y2[i]*(37145.0*z10[i] - 81719.0*z8[i]*r2[i] + 63954.0*z6[i]*r4[i] - 21318.0*z4[i]*r6[i] + 2805.0*z2[i]*r8[i] - 99.0*r10[i]) + 334305.0*z12[i] - 965770.0*z10[i]*r2[i] + 1062347.0*z8[i]*r4[i] - 554268.0*z6[i]*r6[i] + 138567.0*z4[i]*r8[i] - 14586.0*z2[i]*r10[i] + 429.0*r12[i]);
      prCofDY[totalAN*206+i] = 0.135000539051985*y[i]*(-185725.0*z12[i] + 490314.0*z10[i]*r2[i] - 479655.0*z8[i]*r4[i] + 213180.0*z6[i]*r6[i] - 42075.0*z4[i]*r8[i] + 2970.0*z2[i]*r10[i] - 33.0*r12[i]);
      prCofDY[totalAN*207+i] = -0.790482215453542*x[i]*y[i]*z[i]*(37145.0*z10[i] - 81719.0*z8[i]*r2[i] + 63954.0*z6[i]*r4[i] - 21318.0*z4[i]*r6[i] + 2805.0*z2[i]*r8[i] - 99.0*r10[i]);
      prCofDY[totalAN*208+i] = 0.0274050400015396*y[i]*(-334305.0*z12[i] + 817190.0*z10[i]*r2[i] - 735471.0*z8[i]*r4[i] + 298452.0*z6[i]*r6[i] - 53295.0*z4[i]*r8[i] + 3366.0*z2[i]*r10[i] - 22.0*(x2[i] - y2[i])*(37145.0*z10[i] - 66861.0*z8[i]*r2[i] + 40698.0*z6[i]*r4[i] - 9690.0*z4[i]*r6[i] + 765.0*z2[i]*r8[i] - 9.0*r10[i]) - 33.0*r12[i]);
      prCofDY[totalAN*209+i] = 0.130474087834899*x[i]*y[i]*z[i]*(-176985.0*z10[i] + 360525.0*z8[i]*r2[i] - 259578.0*z6[i]*r4[i] + 79002.0*z4[i]*r6[i] - 9405.0*z2[i]*r8[i] - 11.0*(x2[i] - 3.0*y2[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]) + 297.0*r10[i]);
      prCofDY[totalAN*210+i] = -0.917966576744097*y[i]*(2.0*(3.0*x2[i] - y2[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]) + (x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDY[totalAN*211+i] = -5.061318418652*x[i]*y[i]*z[i]*(5.0*(x2[i] - y2[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]) + 2.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]));
      prCofDY[totalAN*212+i] = -0.943121003323134*y[i]*(3.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]) + 4.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]));
      prCofDY[totalAN*213+i] = -12.224245339519*x[i]*y[i]*z[i]*((3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]) + (115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i]));
      prCofDY[totalAN*214+i] = -0.985057761246463*y[i]*((575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]) + 4.0*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i]));
      prCofDY[totalAN*215+i] = 7.71453904240792*x[i]*y[i]*z[i]*((3.0*x2[i] + 3.0*y2[i] - 22.0*z2[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) - 18.0*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]));
      prCofDY[totalAN*216+i] = 1.05635676357028*y[i]*(2.0*(x2[i] + y2[i] - 24.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) - 5.0*(225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDY[totalAN*217+i] = 10.5635676357028*x[i]*y[i]*z[i]*(-x10[i] + 55.0*x8[i]*y2[i] - 330.0*x6[i]*y4[i] + 462.0*x4[i]*y6[i] - 165.0*x2[i]*y8[i] + 11.0*y10[i] - 11.0*(-x2[i] - y2[i] + 8.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDY[totalAN*218+i] = 1.19608850325394*y[i]*(-x12[i] + 66.0*x10[i]*y2[i] - 495.0*x8[i]*y4[i] + 924.0*x6[i]*y6[i] - 495.0*x4[i]*y8[i] + 66.0*x2[i]*y10[i] - y12[i] + 6.0*(x2[i] + y2[i] - 26.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDY[totalAN*219+i] = -228.52490857415*x[i]*y[i]*z[i]*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      prCofDY[totalAN*220+i] = 11.6273091629033*y[i]*(-13.0*x12[i] + 286.0*x10[i]*y2[i] - 1287.0*x8[i]*y4[i] + 1716.0*x6[i]*y6[i] - 715.0*x4[i]*y8[i] + 78.0*x2[i]*y10[i] - y12[i]);

      prCofDZ[totalAN*192+i] = 0;
      prCofDZ[totalAN*193+i] = 4.39470978027212*y[i]*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*194+i] = 124.39320433841*x[i]*y[i]*z[i]*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      prCofDZ[totalAN*195+i] = -5.2817838178514*y[i]*(x2[i] + y2[i] - 24.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*196+i] = 33.803416434249*x[i]*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 22.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      prCofDZ[totalAN*197+i] = 1.92863476060198*y[i]*(575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*198+i] = 173.370165979377*x[i]*y[i]*z[i]*(115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]);
      prCofDZ[totalAN*199+i] = 6.11212266975949*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]);
      prCofDZ[totalAN*200+i] = 30.1798721063403*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDZ[totalAN*201+i] = 1.265329604663*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i]);
      prCofDZ[totalAN*202+i] = 7.34373261395277*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]);
      prCofDZ[totalAN*203+i] = 0.717607483091946*y[i]*(3.0*x2[i] - y2[i])*(37145.0*z10[i] - 66861.0*z8[i]*r2[i] + 40698.0*z6[i]*r4[i] - 9690.0*z4[i]*r6[i] + 765.0*z2[i]*r8[i] - 9.0*r10[i]);
      prCofDZ[totalAN*204+i] = 1.75392256009853*x[i]*y[i]*z[i]*(37145.0*z10[i] - 81719.0*z8[i]*r2[i] + 63954.0*z6[i]*r4[i] - 21318.0*z4[i]*r6[i] + 2805.0*z2[i]*r8[i] - 99.0*r10[i]);
      prCofDZ[totalAN*205+i] = 0.197620553863385*y[i]*(185725.0*z12[i] - 490314.0*z10[i]*r2[i] + 479655.0*z8[i]*r4[i] - 213180.0*z6[i]*r6[i] + 42075.0*z4[i]*r8[i] - 2970.0*z2[i]*r10[i] + 33.0*r12[i]);
      prCofDZ[totalAN*206+i] = 0.0207693137003054*z[i]*(1300075.0*z12[i] - 4056234.0*z10[i]*r2[i] + 4849845.0*z8[i]*r4[i] - 2771340.0*z6[i]*r6[i] + 765765.0*z4[i]*r8[i] - 90090.0*z2[i]*r10[i] + 3003.0*r12[i]);
      prCofDZ[totalAN*207+i] = 0.197620553863385*x[i]*(185725.0*z12[i] - 490314.0*z10[i]*r2[i] + 479655.0*z8[i]*r4[i] - 213180.0*z6[i]*r6[i] + 42075.0*z4[i]*r8[i] - 2970.0*z2[i]*r10[i] + 33.0*r12[i]);
      prCofDZ[totalAN*208+i] = 0.876961280049267*z[i]*(x2[i] - y2[i])*(37145.0*z10[i] - 81719.0*z8[i]*r2[i] + 63954.0*z6[i]*r4[i] - 21318.0*z4[i]*r6[i] + 2805.0*z2[i]*r8[i] - 99.0*r10[i]);
      prCofDZ[totalAN*209+i] = 0.717607483091946*x[i]*(x2[i] - 3.0*y2[i])*(37145.0*z10[i] - 66861.0*z8[i]*r2[i] + 40698.0*z6[i]*r4[i] - 9690.0*z4[i]*r6[i] + 765.0*z2[i]*r8[i] - 9.0*r10[i]);
      prCofDZ[totalAN*210+i] = 1.83593315348819*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10925.0*z8[i] - 15732.0*z6[i]*r2[i] + 7182.0*z4[i]*r4[i] - 1140.0*z2[i]*r6[i] + 45.0*r8[i]);
      prCofDZ[totalAN*211+i] = 1.265329604663*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(10925.0*z8[i] - 12236.0*z6[i]*r2[i] + 3990.0*z4[i]*r4[i] - 380.0*z2[i]*r6[i] + 5.0*r8[i]);
      prCofDZ[totalAN*212+i] = 15.0899360531701*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(575.0*z6[i] - 483.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - 5.0*r6[i]);
      prCofDZ[totalAN*213+i] = 6.11212266975949*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(805.0*z6[i] - 483.0*z4[i]*r2[i] + 63.0*z2[i]*r4[i] - r6[i]);
      prCofDZ[totalAN*214+i] = 21.6712707474222*z[i]*(115.0*z4[i] - 46.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*215+i] = 1.92863476060198*x[i]*(575.0*z4[i] - 138.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      prCofDZ[totalAN*216+i] = 16.9017082171245*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 22.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*217+i] = -5.2817838178514*x[i]*(x2[i] + y2[i] - 24.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      prCofDZ[totalAN*218+i] = 31.0983010846025*z[i]*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*219+i] = 4.39470978027212*x[i]*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      prCofDZ[totalAN*220+i] = 0;
        }
    if (lMax > 14){ 
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

        if(return_derivatives){
      prCofDX[totalAN*221+i] = 25.3275195257212*x[i]*y[i]*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      prCofDX[totalAN*222+i] = 64.7381175928202*y[i]*z[i]*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      prCofDX[totalAN*223+i] = 1.21436161302378*x[i]*y[i]*(-13.0*x12[i] + 286.0*x10[i]*y2[i] - 1287.0*x8[i]*y4[i] + 1716.0*x6[i]*y6[i] - 715.0*x4[i]*y8[i] + 78.0*x2[i]*y10[i] - y12[i] - 26.0*(x2[i] + y2[i] - 28.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]));
      prCofDX[totalAN*224+i] = 22.2596160509206*y[i]*z[i]*(x2[i]*(-6.0*x10[i] + 110.0*x8[i]*y2[i] - 396.0*x6[i]*y4[i] + 396.0*x4[i]*y6[i] - 110.0*x2[i]*y8[i] + 6.0*y10[i]) + (-3.0*x2[i] - 3.0*y2[i] + 26.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDX[totalAN*225+i] = 1.07096627658806*x[i]*y[i]*(-2.0*(-x2[i] - y2[i] + 26.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 11.0*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDX[totalAN*226+i] = 12.2108942967563*y[i]*z[i]*(-4.0*x2[i]*(-x2[i] - y2[i] + 8.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + (261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*227+i] = 2.99104603301142*x[i]*y[i]*(-(225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) + 12.0*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i]));
      prCofDX[totalAN*228+i] = 7.38445476455181*y[i]*z[i]*(-14.0*x2[i]*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i]));
      prCofDX[totalAN*229+i] = 0.95268114867001*x[i]*y[i]*((3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]) - 4.0*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i]));
      prCofDX[totalAN*230+i] = 1.91505886020364*y[i]*z[i]*(-8.0*x2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]) + (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]));
      prCofDX[totalAN*231+i] = 4.62530657238986*x[i]*y[i]*(2.0*(x2[i] - y2[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]) - (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]));
      prCofDX[totalAN*232+i] = 2.49470484396446*y[i]*z[i]*(-22.0*x2[i]*(x2[i] - y2[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]) + (3.0*x2[i] - y2[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]));
      prCofDX[totalAN*233+i] = 0.247823796536837*x[i]*y[i]*(190095.0*z12[i] - 432630.0*z10[i]*r2[i] + 360525.0*z8[i]*r4[i] - 134596.0*z6[i]*r6[i] + 21945.0*z4[i]*r8[i] - 1254.0*z2[i]*r10[i] - 22.0*(3.0*x2[i] - y2[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]) + 11.0*r12[i]);
      prCofDX[totalAN*234+i] = 0.0324014967813841*y[i]*z[i]*(-52.0*x2[i]*(58995.0*z10[i] - 120175.0*z8[i]*r2[i] + 86526.0*z6[i]*r4[i] - 26334.0*z4[i]*r6[i] + 3135.0*z2[i]*r8[i] - 99.0*r10[i]) + 570285.0*z12[i] - 1533870.0*z10[i]*r2[i] + 1562275.0*z8[i]*r4[i] - 749892.0*z6[i]*r6[i] + 171171.0*z4[i]*r8[i] - 16302.0*z2[i]*r10[i] + 429.0*r12[i]);
      prCofDX[totalAN*235+i] = -0.191125214847445*x[i]*y[i]*(334305.0*z12[i] - 817190.0*z10[i]*r2[i] + 735471.0*z8[i]*r4[i] - 298452.0*z6[i]*r6[i] + 53295.0*z4[i]*r8[i] - 3366.0*z2[i]*r10[i] + 33.0*r12[i]);
      prCofDX[totalAN*236+i] = 0.161051679199949*x[i]*z[i]*(-334305.0*z12[i] + 965770.0*z10[i]*r2[i] - 1062347.0*z8[i]*r4[i] + 554268.0*z6[i]*r6[i] - 138567.0*z4[i]*r8[i] + 14586.0*z2[i]*r10[i] - 429.0*r12[i]);
      prCofDX[totalAN*237+i] = -0.191125214847445*x2[i]*(334305.0*z12[i] - 817190.0*z10[i]*r2[i] + 735471.0*z8[i]*r4[i] - 298452.0*z6[i]*r6[i] + 53295.0*z4[i]*r8[i] - 3366.0*z2[i]*r10[i] + 33.0*r12[i]) + 10180.9304040532*z14[i] - 31947.0574747876*z12[i]*r2[i] + 39046.403580296*z10[i]*r4[i] - 23427.8421481776*z8[i]*r6[i] + 7130.21282770622*z6[i]*r8[i] - 1018.60183252946*z4[i]*r10[i] + 53.6106227647084*z2[i]*r12[i] - 0.45050943499755*r14[i];
      prCofDX[totalAN*238+i] = 0.0324014967813841*x[i]*z[i]*(570285.0*z12[i] - 1533870.0*z10[i]*r2[i] + 1562275.0*z8[i]*r4[i] - 749892.0*z6[i]*r6[i] + 171171.0*z4[i]*r8[i] - 16302.0*z2[i]*r10[i] - 26.0*(x2[i] - y2[i])*(58995.0*z10[i] - 120175.0*z8[i]*r2[i] + 86526.0*z6[i]*r4[i] - 26334.0*z4[i]*r6[i] + 3135.0*z2[i]*r8[i] - 99.0*r10[i]) + 429.0*r12[i]);
      prCofDX[totalAN*239+i] = -5.45212352381041*x2[i]*(x2[i] - 3.0*y2[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]) + 0.123911898268418*(x2[i] - y2[i])*(190095.0*z12[i] - 432630.0*z10[i]*r2[i] + 360525.0*z8[i]*r4[i] - 134596.0*z6[i]*r6[i] + 21945.0*z4[i]*r8[i] - 1254.0*z2[i]*r10[i] + 11.0*r12[i]);
      prCofDX[totalAN*240+i] = 1.24735242198223*x[i]*z[i]*(2.0*(x2[i] - 3.0*y2[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]) - 11.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDX[totalAN*241+i] = -4.62530657238986*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]) + 2.31265328619493*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]);
      prCofDX[totalAN*242+i] = 1.91505886020364*x[i]*z[i]*((x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]) - 4.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]));
      prCofDX[totalAN*243+i] = -3.81072459468004*x2[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i]) + 0.476340574335005*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]);
      prCofDX[totalAN*244+i] = 1.84611369113795*x[i]*z[i]*(-7.0*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]) + 4.0*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i]));
      prCofDX[totalAN*245+i] = -2.99104603301142*x2[i]*(225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) + 4.48656904951713*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDX[totalAN*246+i] = 12.2108942967563*x[i]*z[i]*(-2.0*(-x2[i] - y2[i] + 8.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) + (261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDX[totalAN*247+i] = -2.14193255317612*x2[i]*(-x2[i] - y2[i] + 26.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) + 5.89031452123433*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDX[totalAN*248+i] = 11.1298080254603*x[i]*z[i]*(-x12[i] + 66.0*x10[i]*y2[i] - 495.0*x8[i]*y4[i] + 924.0*x6[i]*y6[i] - 495.0*x4[i]*y8[i] + 66.0*x2[i]*y10[i] - y12[i] + 2.0*(-3.0*x2[i] - 3.0*y2[i] + 26.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDX[totalAN*249+i] = -1.21436161302378*x2[i]*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]) - 7.89335048465457*(x2[i] + y2[i] - 28.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDX[totalAN*250+i] = 64.7381175928202*x[i]*z[i]*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      prCofDX[totalAN*251+i] = 12.6637597628606*x14[i] - 1152.40213842031*x12[i]*y2[i] + 12676.4235226235*x10[i]*y4[i] - 38029.2705678703*x8[i]*y6[i] + 38029.2705678703*x6[i]*y8[i] - 12676.4235226235*x4[i]*y10[i] + 1152.40213842031*x2[i]*y12[i] - 12.6637597628606*y14[i];

      prCofDY[totalAN*221+i] = 12.6637597628606*x14[i] - 1152.40213842031*x12[i]*y2[i] + 12676.4235226235*x10[i]*y4[i] - 38029.2705678703*x8[i]*y6[i] + 38029.2705678703*x6[i]*y8[i] - 12676.4235226235*x4[i]*y10[i] + 1152.40213842031*x2[i]*y12[i] - 12.6637597628606*y14[i];
      prCofDY[totalAN*222+i] = 64.7381175928202*x[i]*z[i]*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      prCofDY[totalAN*223+i] = -1.21436161302378*y2[i]*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]) - 7.89335048465457*(x2[i] + y2[i] - 28.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDY[totalAN*224+i] = 22.2596160509206*x[i]*z[i]*(y2[i]*(-6.0*x10[i] + 110.0*x8[i]*y2[i] - 396.0*x6[i]*y4[i] + 396.0*x4[i]*y6[i] - 110.0*x2[i]*y8[i] + 6.0*y10[i]) + (-3.0*x2[i] - 3.0*y2[i] + 26.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDY[totalAN*225+i] = -2.14193255317612*y2[i]*(-x2[i] - y2[i] + 26.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 5.89031452123433*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDY[totalAN*226+i] = 12.2108942967563*x[i]*z[i]*(-4.0*y2[i]*(-x2[i] - y2[i] + 8.0*z2[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + (261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDY[totalAN*227+i] = -2.99104603301142*y2[i]*(225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) + 4.48656904951713*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDY[totalAN*228+i] = 7.38445476455181*x[i]*z[i]*(-14.0*y2[i]*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i]) + (x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i]));
      prCofDY[totalAN*229+i] = -3.81072459468004*y2[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i]) + 0.476340574335005*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]);
      prCofDY[totalAN*230+i] = 1.91505886020364*x[i]*z[i]*(-8.0*y2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]) + (x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]));
      prCofDY[totalAN*231+i] = -4.62530657238986*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]) + 2.31265328619493*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]);
      prCofDY[totalAN*232+i] = 2.49470484396446*x[i]*z[i]*(-22.0*y2[i]*(x2[i] - y2[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]) + (x2[i] - 3.0*y2[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]));
      prCofDY[totalAN*233+i] = -5.45212352381041*y2[i]*(3.0*x2[i] - y2[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]) + 0.123911898268418*(x2[i] - y2[i])*(190095.0*z12[i] - 432630.0*z10[i]*r2[i] + 360525.0*z8[i]*r4[i] - 134596.0*z6[i]*r6[i] + 21945.0*z4[i]*r8[i] - 1254.0*z2[i]*r10[i] + 11.0*r12[i]);
      prCofDY[totalAN*234+i] = 0.0324014967813841*x[i]*z[i]*(-52.0*y2[i]*(58995.0*z10[i] - 120175.0*z8[i]*r2[i] + 86526.0*z6[i]*r4[i] - 26334.0*z4[i]*r6[i] + 3135.0*z2[i]*r8[i] - 99.0*r10[i]) + 570285.0*z12[i] - 1533870.0*z10[i]*r2[i] + 1562275.0*z8[i]*r4[i] - 749892.0*z6[i]*r6[i] + 171171.0*z4[i]*r8[i] - 16302.0*z2[i]*r10[i] + 429.0*r12[i]);
      prCofDY[totalAN*235+i] = -0.191125214847445*y2[i]*(334305.0*z12[i] - 817190.0*z10[i]*r2[i] + 735471.0*z8[i]*r4[i] - 298452.0*z6[i]*r6[i] + 53295.0*z4[i]*r8[i] - 3366.0*z2[i]*r10[i] + 33.0*r12[i]) + 10180.9304040532*z14[i] - 31947.0574747876*z12[i]*r2[i] + 39046.403580296*z10[i]*r4[i] - 23427.8421481776*z8[i]*r6[i] + 7130.21282770622*z6[i]*r8[i] - 1018.60183252946*z4[i]*r10[i] + 53.6106227647084*z2[i]*r12[i] - 0.45050943499755*r14[i];
      prCofDY[totalAN*236+i] = 0.161051679199949*y[i]*z[i]*(-334305.0*z12[i] + 965770.0*z10[i]*r2[i] - 1062347.0*z8[i]*r4[i] + 554268.0*z6[i]*r6[i] - 138567.0*z4[i]*r8[i] + 14586.0*z2[i]*r10[i] - 429.0*r12[i]);
      prCofDY[totalAN*237+i] = -0.191125214847445*x[i]*y[i]*(334305.0*z12[i] - 817190.0*z10[i]*r2[i] + 735471.0*z8[i]*r4[i] - 298452.0*z6[i]*r6[i] + 53295.0*z4[i]*r8[i] - 3366.0*z2[i]*r10[i] + 33.0*r12[i]);
      prCofDY[totalAN*238+i] = 0.0324014967813841*y[i]*z[i]*(-570285.0*z12[i] + 1533870.0*z10[i]*r2[i] - 1562275.0*z8[i]*r4[i] + 749892.0*z6[i]*r6[i] - 171171.0*z4[i]*r8[i] + 16302.0*z2[i]*r10[i] - 26.0*(x2[i] - y2[i])*(58995.0*z10[i] - 120175.0*z8[i]*r2[i] + 86526.0*z6[i]*r4[i] - 26334.0*z4[i]*r6[i] + 3135.0*z2[i]*r8[i] - 99.0*r10[i]) - 429.0*r12[i]);
      prCofDY[totalAN*239+i] = 0.247823796536837*x[i]*y[i]*(-190095.0*z12[i] + 432630.0*z10[i]*r2[i] - 360525.0*z8[i]*r4[i] + 134596.0*z6[i]*r6[i] - 21945.0*z4[i]*r8[i] + 1254.0*z2[i]*r10[i] - 22.0*(x2[i] - 3.0*y2[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]) - 11.0*r12[i]);
      prCofDY[totalAN*240+i] = -1.24735242198223*y[i]*z[i]*(2.0*(3.0*x2[i] - y2[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]) + 11.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDY[totalAN*241+i] = -4.62530657238986*x[i]*y[i]*(2.0*(x2[i] - y2[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]) + (x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]));
      prCofDY[totalAN*242+i] = -1.91505886020364*y[i]*z[i]*((5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]) + 4.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]));
      prCofDY[totalAN*243+i] = -0.95268114867001*x[i]*y[i]*((3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]) + 4.0*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i]));
      prCofDY[totalAN*244+i] = -1.84611369113795*y[i]*z[i]*(7.0*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]) + 4.0*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i]));
      prCofDY[totalAN*245+i] = -2.99104603301142*x[i]*y[i]*((225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) + 12.0*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i]));
      prCofDY[totalAN*246+i] = 12.2108942967563*y[i]*z[i]*(2.0*(x2[i] + y2[i] - 8.0*z2[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) - (261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDY[totalAN*247+i] = 1.07096627658806*x[i]*y[i]*(2.0*(x2[i] + y2[i] - 26.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) - 11.0*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDY[totalAN*248+i] = 11.1298080254603*y[i]*z[i]*(-x12[i] + 66.0*x10[i]*y2[i] - 495.0*x8[i]*y4[i] + 924.0*x6[i]*y6[i] - 495.0*x4[i]*y8[i] + 66.0*x2[i]*y10[i] - y12[i] - 2.0*(-3.0*x2[i] - 3.0*y2[i] + 26.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDY[totalAN*249+i] = 1.21436161302378*x[i]*y[i]*(-x12[i] + 78.0*x10[i]*y2[i] - 715.0*x8[i]*y4[i] + 1716.0*x6[i]*y6[i] - 1287.0*x4[i]*y8[i] + 286.0*x2[i]*y10[i] - 13.0*y12[i] + 26.0*(x2[i] + y2[i] - 28.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]));
      prCofDY[totalAN*250+i] = -64.7381175928202*y[i]*z[i]*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      prCofDY[totalAN*251+i] = 25.3275195257212*x[i]*y[i]*(-7.0*x12[i] + 182.0*x10[i]*y2[i] - 1001.0*x8[i]*y4[i] + 1716.0*x6[i]*y6[i] - 1001.0*x4[i]*y8[i] + 182.0*x2[i]*y10[i] - 7.0*y12[i]);

      prCofDZ[totalAN*221+i] = 0;
      prCofDZ[totalAN*222+i] = 9.24830251326002*x[i]*y[i]*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      prCofDZ[totalAN*223+i] = 34.0021251646659*y[i]*z[i]*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*224+i] = -22.2596160509206*x[i]*y[i]*(x2[i] + y2[i] - 26.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      prCofDZ[totalAN*225+i] = 55.6902463825791*y[i]*z[i]*(-x2[i] - y2[i] + 8.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*226+i] = 12.2108942967563*x[i]*y[i]*(225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      prCofDZ[totalAN*227+i] = 23.9283682640914*y[i]*z[i]*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*228+i] = 51.6911833518627*x[i]*y[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i]);
      prCofDZ[totalAN*229+i] = 11.9765630118516*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]);
      prCofDZ[totalAN*230+i] = 13.4054120214255*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]);
      prCofDZ[totalAN*231+i] = 18.5012262895594*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]);
      prCofDZ[totalAN*232+i] = 27.441753283609*x[i]*y[i]*(x2[i] - y2[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]);
      prCofDZ[totalAN*233+i] = 0.991295186147348*y[i]*z[i]*(3.0*x2[i] - y2[i])*(58995.0*z10[i] - 120175.0*z8[i]*r2[i] + 86526.0*z6[i]*r4[i] - 26334.0*z4[i]*r6[i] + 3135.0*z2[i]*r8[i] - 99.0*r10[i]);
      prCofDZ[totalAN*234+i] = 0.421219458157993*x[i]*y[i]*(334305.0*z12[i] - 817190.0*z10[i]*r2[i] + 735471.0*z8[i]*r4[i] - 298452.0*z6[i]*r6[i] + 53295.0*z4[i]*r8[i] - 3366.0*z2[i]*r10[i] + 33.0*r12[i]);
      prCofDZ[totalAN*235+i] = 0.235231033658394*y[i]*z[i]*(334305.0*z12[i] - 965770.0*z10[i]*r2[i] + 1062347.0*z8[i]*r4[i] - 554268.0*z6[i]*r6[i] + 138567.0*z4[i]*r8[i] - 14586.0*z2[i]*r10[i] + 429.0*r12[i]);
      prCofDZ[totalAN*236+i] = 57686.1231588633*z14[i] - 194423.600276169*z12[i]*r2[i] + 256639.152364543*z10[i]*r4[i] - 167373.360237745*z8[i]*r6[i] + 55791.1200792485*z6[i]*r8[i] - 8809.12422303923*z4[i]*r10[i] + 518.183777825837*z2[i]*r12[i] - 4.93508359834131*r14[i];
      prCofDZ[totalAN*237+i] = 0.235231033658394*x[i]*z[i]*(334305.0*z12[i] - 965770.0*z10[i]*r2[i] + 1062347.0*z8[i]*r4[i] - 554268.0*z6[i]*r6[i] + 138567.0*z4[i]*r8[i] - 14586.0*z2[i]*r10[i] + 429.0*r12[i]);
      prCofDZ[totalAN*238+i] = 0.210609729078997*(x2[i] - y2[i])*(334305.0*z12[i] - 817190.0*z10[i]*r2[i] + 735471.0*z8[i]*r4[i] - 298452.0*z6[i]*r6[i] + 53295.0*z4[i]*r8[i] - 3366.0*z2[i]*r10[i] + 33.0*r12[i]);
      prCofDZ[totalAN*239+i] = 0.991295186147348*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(58995.0*z10[i] - 120175.0*z8[i]*r2[i] + 86526.0*z6[i]*r4[i] - 26334.0*z4[i]*r6[i] + 3135.0*z2[i]*r8[i] - 99.0*r10[i]);
      prCofDZ[totalAN*240+i] = 6.86043832090226*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(6555.0*z10[i] - 10925.0*z8[i]*r2[i] + 6118.0*z6[i]*r4[i] - 1330.0*z4[i]*r6[i] + 95.0*z2[i]*r8[i] - r10[i]);
      prCofDZ[totalAN*241+i] = 18.5012262895594*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(1725.0*z8[i] - 2300.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 140.0*z2[i]*r6[i] + 5.0*r8[i]);
      prCofDZ[totalAN*242+i] = 6.70270601071276*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(3105.0*z8[i] - 3220.0*z6[i]*r2[i] + 966.0*z4[i]*r4[i] - 84.0*z2[i]*r6[i] + r8[i]);
      prCofDZ[totalAN*243+i] = 11.9765630118516*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(1035.0*z6[i] - 805.0*z4[i]*r2[i] + 161.0*z2[i]*r4[i] - 7.0*r6[i]);
      prCofDZ[totalAN*244+i] = 6.46139791898283*(1035.0*z6[i] - 575.0*z4[i]*r2[i] + 69.0*z2[i]*r4[i] - r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*245+i] = 23.9283682640914*x[i]*z[i]*(135.0*z4[i] - 50.0*z2[i]*r2[i] + 3.0*r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      prCofDZ[totalAN*246+i] = 6.10544714837816*(225.0*z4[i] - 50.0*z2[i]*r2[i] + r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*247+i] = 55.6902463825791*x[i]*z[i]*(-x2[i] - y2[i] + 8.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      prCofDZ[totalAN*248+i] = -5.56490401273015*(x2[i] + y2[i] - 26.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*249+i] = 34.0021251646659*x[i]*z[i]*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      prCofDZ[totalAN*250+i] = 4.62415125663001*x14[i] - 420.797764353331*x12[i]*y2[i] + 4628.77540788664*x10[i]*y4[i] - 13886.3262236599*x8[i]*y6[i] + 13886.3262236599*x6[i]*y8[i] - 4628.77540788664*x4[i]*y10[i] + 420.797764353331*x2[i]*y12[i] - 4.62415125663001*y14[i];
      prCofDZ[totalAN*251+i] = 0;
        }
    if (lMax > 15){ 
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

        if(return_derivatives){
      prCofDX[totalAN*252+i] = 13.7174494214084*y[i]*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      prCofDX[totalAN*253+i] = 145.49552259692*x[i]*y[i]*z[i]*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      prCofDX[totalAN*254+i] = 1.23186332318453*y[i]*(x2[i]*(-14.0*x12[i] + 364.0*x10[i]*y2[i] - 2002.0*x8[i]*y4[i] + 3432.0*x6[i]*y6[i] - 2002.0*x4[i]*y8[i] + 364.0*x2[i]*y10[i] - 14.0*y12[i]) - 7.0*(x2[i] + y2[i] - 30.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]));
      prCofDX[totalAN*255+i] = 3.89549386728723*x[i]*y[i]*z[i]*(-39.0*x12[i] + 858.0*x10[i]*y2[i] - 3861.0*x8[i]*y4[i] + 5148.0*x6[i]*y6[i] - 2145.0*x4[i]*y8[i] + 234.0*x2[i]*y10[i] - 3.0*y12[i] + 26.0*(-3.0*x2[i] - 3.0*y2[i] + 28.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]));
      prCofDX[totalAN*256+i] = 2.1701251531576*y[i]*(-4.0*x2[i]*(-x2[i] - y2[i] + 28.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]) + (899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDX[totalAN*257+i] = 0.855908903026109*x[i]*y[i]*z[i]*(-10.0*(-3.0*x2[i] - 3.0*y2[i] + 26.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 11.0*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDX[totalAN*258+i] = 1.0086983156795*y[i]*(-6.0*x2[i]*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + (8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*259+i] = 1.16640722205483*x[i]*y[i]*z[i]*(-7.0*(261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) + 12.0*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i]));
      prCofDX[totalAN*260+i] = 0.549849637559955*y[i]*(-56.0*x2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i]) + (7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]));
      prCofDX[totalAN*261+i] = 0.673425523638079*x[i]*y[i]*z[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]) - 12.0*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i]));
      prCofDX[totalAN*262+i] = 0.133213092171985*y[i]*(-10.0*x2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]) + (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]));
      prCofDX[totalAN*263+i] = 0.31398626939213*x[i]*y[i]*z[i]*(2.0*(x2[i] - y2[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]) - 11.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]));
      prCofDX[totalAN*264+i] = 0.166145916780101*y[i]*(-132.0*x2[i]*(x2[i] - y2[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]) + (3.0*x2[i] - y2[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]));
      prCofDX[totalAN*265+i] = 0.309117970363509*x[i]*y[i]*z[i]*(310155.0*z12[i] - 780390.0*z10[i]*r2[i] + 740025.0*z8[i]*r4[i] - 328900.0*z6[i]*r6[i] + 69069.0*z4[i]*r8[i] - 6006.0*z2[i]*r10[i] - 26.0*(3.0*x2[i] - y2[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]) + 143.0*r12[i]);
      prCofDX[totalAN*266+i] = 0.00631774627238717*y[i]*(-182.0*x2[i]*(190095.0*z12[i] - 432630.0*z10[i]*r2[i] + 360525.0*z8[i]*r4[i] - 134596.0*z6[i]*r6[i] + 21945.0*z4[i]*r8[i] - 1254.0*z2[i]*r10[i] + 11.0*r12[i]) + 5892945.0*z14[i] - 17298645.0*z12[i]*r2[i] + 19684665.0*z10[i]*r4[i] - 10935925.0*z8[i]*r6[i] + 3062059.0*z6[i]*r8[i] - 399399.0*z4[i]*r10[i] + 19019.0*z2[i]*r12[i] - 143.0*r14[i]);
      prCofDX[totalAN*267+i] = -0.242226050218644*x[i]*y[i]*z[i]*(570285.0*z12[i] - 1533870.0*z10[i]*r2[i] + 1562275.0*z8[i]*r4[i] - 749892.0*z6[i]*r6[i] + 171171.0*z4[i]*r8[i] - 16302.0*z2[i]*r10[i] + 429.0*r12[i]);
      prCofDX[totalAN*268+i] = 0.0118689785420445*x[i]*(-9694845.0*z14[i] + 30421755.0*z12[i]*r2[i] - 37182145.0*z10[i]*r4[i] + 22309287.0*z8[i]*r6[i] - 6789783.0*z6[i]*r8[i] + 969969.0*z4[i]*r10[i] - 51051.0*z2[i]*r12[i] + 429.0*r14[i]);
      prCofDX[totalAN*269+i] = 0.00115345738199354*z[i]*(-210.0*x2[i]*(570285.0*z12[i] - 1533870.0*z10[i]*r2[i] + 1562275.0*z8[i]*r4[i] - 749892.0*z6[i]*r6[i] + 171171.0*z4[i]*r8[i] - 16302.0*z2[i]*r10[i] + 429.0*r12[i]) + 17678835.0*z14[i] - 59879925.0*z12[i]*r2[i] + 80528175.0*z10[i]*r4[i] - 54679625.0*z8[i]*r6[i] + 19684665.0*z6[i]*r8[i] - 3594591.0*z4[i]*r10[i] + 285285.0*z2[i]*r12[i] - 6435.0*r14[i]);
      prCofDX[totalAN*270+i] = 0.00631774627238717*x[i]*(5892945.0*z14[i] - 17298645.0*z12[i]*r2[i] + 19684665.0*z10[i]*r4[i] - 10935925.0*z8[i]*r6[i] + 3062059.0*z6[i]*r8[i] - 399399.0*z4[i]*r10[i] + 19019.0*z2[i]*r12[i] - 91.0*(x2[i] - y2[i])*(190095.0*z12[i] - 432630.0*z10[i]*r2[i] + 360525.0*z8[i]*r4[i] - 134596.0*z6[i]*r6[i] + 21945.0*z4[i]*r8[i] - 1254.0*z2[i]*r10[i] + 11.0*r12[i]) - 143.0*r14[i]);
      prCofDX[totalAN*271+i] = 0.154558985181755*z[i]*(-52.0*x2[i]*(x2[i] - 3.0*y2[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]) + (x2[i] - y2[i])*(310155.0*z12[i] - 780390.0*z10[i]*r2[i] + 740025.0*z8[i]*r4[i] - 328900.0*z6[i]*r6[i] + 69069.0*z4[i]*r8[i] - 6006.0*z2[i]*r10[i] + 143.0*r12[i]));
      prCofDX[totalAN*272+i] = 0.166145916780101*x[i]*((x2[i] - 3.0*y2[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]) - 33.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]));
      prCofDX[totalAN*273+i] = 0.156993134696065*z[i]*(-22.0*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]) + (x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]));
      prCofDX[totalAN*274+i] = 0.133213092171985*x[i]*((x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]) - 5.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]));
      prCofDX[totalAN*275+i] = 0.336712761819039*z[i]*(-24.0*x2[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i]) + 7.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]));
      prCofDX[totalAN*276+i] = 0.549849637559955*x[i]*((x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]) - 7.0*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*277+i] = 0.583203611027414*z[i]*(-14.0*x2[i]*(261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) + 3.0*(8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*278+i] = 1.0086983156795*x[i]*(-3.0*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) + (8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDX[totalAN*279+i] = 0.427954451513054*z[i]*(-20.0*x2[i]*(-3.0*x2[i] - 3.0*y2[i] + 26.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) + 11.0*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]));
      prCofDX[totalAN*280+i] = 2.1701251531576*x[i]*(-(-x2[i] - y2[i] + 28.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]) + (899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDX[totalAN*281+i] = 1.94774693364361*z[i]*(-6.0*x2[i]*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]) + 13.0*(-3.0*x2[i] - 3.0*y2[i] + 28.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]));
      prCofDX[totalAN*282+i] = 1.23186332318453*x[i]*(-x14[i] + 91.0*x12[i]*y2[i] - 1001.0*x10[i]*y4[i] + 3003.0*x8[i]*y6[i] - 3003.0*x6[i]*y8[i] + 1001.0*x4[i]*y10[i] - 91.0*x2[i]*y12[i] + y14[i] - 7.0*(x2[i] + y2[i] - 30.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]));
      prCofDX[totalAN*283+i] = 72.7477612984602*z[i]*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      prCofDX[totalAN*284+i] = 13.7174494214084*x[i]*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);

      prCofDY[totalAN*252+i] =13.7174494214084*x[i]*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
      prCofDY[totalAN*253+i] = 72.7477612984602*z[i]*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      prCofDY[totalAN*254+i] = 1.23186332318453*x[i]*(y2[i]*(-14.0*x12[i] + 364.0*x10[i]*y2[i] - 2002.0*x8[i]*y4[i] + 3432.0*x6[i]*y6[i] - 2002.0*x4[i]*y8[i] + 364.0*x2[i]*y10[i] - 14.0*y12[i]) - 7.0*(x2[i] + y2[i] - 30.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]));
      prCofDY[totalAN*255+i] = 1.94774693364361*z[i]*(-6.0*y2[i]*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]) + 13.0*(-3.0*x2[i] - 3.0*y2[i] + 28.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]));
      prCofDY[totalAN*256+i] = 2.1701251531576*x[i]*(-4.0*y2[i]*(-x2[i] - y2[i] + 28.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]) + (899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDY[totalAN*257+i] = 0.427954451513054*z[i]*(-20.0*y2[i]*(-3.0*x2[i] - 3.0*y2[i] + 26.0*z2[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 11.0*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]));
      prCofDY[totalAN*258+i] = 1.0086983156795*x[i]*(-6.0*y2[i]*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + (8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDY[totalAN*259+i] = 0.583203611027414*z[i]*(-14.0*y2[i]*(261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) + 3.0*(8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]));
      prCofDY[totalAN*260+i] = 0.549849637559955*x[i]*(-56.0*y2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i]) + (x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]));
      prCofDY[totalAN*261+i] = 0.336712761819039*z[i]*(-24.0*y2[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i]) + 7.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]));
      prCofDY[totalAN*262+i] = 0.133213092171985*x[i]*(-10.0*y2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]) + (x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]));
      prCofDY[totalAN*263+i] = 0.156993134696065*z[i]*(-22.0*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]) + (x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]));
      prCofDY[totalAN*264+i] = 0.166145916780101*x[i]*(-132.0*y2[i]*(x2[i] - y2[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]) + (x2[i] - 3.0*y2[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]));
      prCofDY[totalAN*265+i] = 0.154558985181755*z[i]*(-52.0*y2[i]*(3.0*x2[i] - y2[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]) + (x2[i] - y2[i])*(310155.0*z12[i] - 780390.0*z10[i]*r2[i] + 740025.0*z8[i]*r4[i] - 328900.0*z6[i]*r6[i] + 69069.0*z4[i]*r8[i] - 6006.0*z2[i]*r10[i] + 143.0*r12[i]));
      prCofDY[totalAN*266+i] = 0.00631774627238717*x[i]*(-182.0*y2[i]*(190095.0*z12[i] - 432630.0*z10[i]*r2[i] + 360525.0*z8[i]*r4[i] - 134596.0*z6[i]*r6[i] + 21945.0*z4[i]*r8[i] - 1254.0*z2[i]*r10[i] + 11.0*r12[i]) + 5892945.0*z14[i] - 17298645.0*z12[i]*r2[i] + 19684665.0*z10[i]*r4[i] - 10935925.0*z8[i]*r6[i] + 3062059.0*z6[i]*r8[i] - 399399.0*z4[i]*r10[i] + 19019.0*z2[i]*r12[i] - 143.0*r14[i]);
      prCofDY[totalAN*267+i] = 0.00115345738199354*z[i]*(-210.0*y2[i]*(570285.0*z12[i] - 1533870.0*z10[i]*r2[i] + 1562275.0*z8[i]*r4[i] - 749892.0*z6[i]*r6[i] + 171171.0*z4[i]*r8[i] - 16302.0*z2[i]*r10[i] + 429.0*r12[i]) + 17678835.0*z14[i] - 59879925.0*z12[i]*r2[i] + 80528175.0*z10[i]*r4[i] - 54679625.0*z8[i]*r6[i] + 19684665.0*z6[i]*r8[i] - 3594591.0*z4[i]*r10[i] + 285285.0*z2[i]*r12[i] - 6435.0*r14[i]);
      prCofDY[totalAN*268+i] = 0.0118689785420445*y[i]*(-9694845.0*z14[i] + 30421755.0*z12[i]*r2[i] - 37182145.0*z10[i]*r4[i] + 22309287.0*z8[i]*r6[i] - 6789783.0*z6[i]*r8[i] + 969969.0*z4[i]*r10[i] - 51051.0*z2[i]*r12[i] + 429.0*r14[i]);
      prCofDY[totalAN*269+i] = -0.242226050218644*x[i]*y[i]*z[i]*(570285.0*z12[i] - 1533870.0*z10[i]*r2[i] + 1562275.0*z8[i]*r4[i] - 749892.0*z6[i]*r6[i] + 171171.0*z4[i]*r8[i] - 16302.0*z2[i]*r10[i] + 429.0*r12[i]);
      prCofDY[totalAN*270+i] = 0.00631774627238717*y[i]*(-5892945.0*z14[i] + 17298645.0*z12[i]*r2[i] - 19684665.0*z10[i]*r4[i] + 10935925.0*z8[i]*r6[i] - 3062059.0*z6[i]*r8[i] + 399399.0*z4[i]*r10[i] - 19019.0*z2[i]*r12[i] - 91.0*(x2[i] - y2[i])*(190095.0*z12[i] - 432630.0*z10[i]*r2[i] + 360525.0*z8[i]*r4[i] - 134596.0*z6[i]*r6[i] + 21945.0*z4[i]*r8[i] - 1254.0*z2[i]*r10[i] + 11.0*r12[i]) + 143.0*r14[i]);
      prCofDY[totalAN*271+i] = 0.309117970363509*x[i]*y[i]*z[i]*(-310155.0*z12[i] + 780390.0*z10[i]*r2[i] - 740025.0*z8[i]*r4[i] + 328900.0*z6[i]*r6[i] - 69069.0*z4[i]*r8[i] + 6006.0*z2[i]*r10[i] - 26.0*(x2[i] - 3.0*y2[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]) - 143.0*r12[i]);
      prCofDY[totalAN*272+i] = -0.166145916780101*y[i]*((3.0*x2[i] - y2[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]) + 33.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]));
      prCofDY[totalAN*273+i] = -0.31398626939213*x[i]*y[i]*z[i]*(2.0*(x2[i] - y2[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]) + 11.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]));
      prCofDY[totalAN*274+i] = -0.133213092171985*y[i]*((5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]) + 5.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]));
      prCofDY[totalAN*275+i] = -0.673425523638079*x[i]*y[i]*z[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]) + 12.0*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i]));
      prCofDY[totalAN*276+i] = -0.549849637559955*y[i]*((7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]) + 7.0*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]));
      prCofDY[totalAN*277+i] = -1.16640722205483*x[i]*y[i]*z[i]*(7.0*(261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) + 12.0*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i]));
      prCofDY[totalAN*278+i] = -1.0086983156795*y[i]*(3.0*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) + (8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDY[totalAN*279+i] = 0.855908903026109*x[i]*y[i]*z[i]*(10.0*(3.0*x2[i] + 3.0*y2[i] - 26.0*z2[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) - 11.0*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDY[totalAN*280+i] = 2.1701251531576*y[i]*((x2[i] + y2[i] - 28.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]) - (899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDY[totalAN*281+i] = 3.89549386728723*x[i]*y[i]*z[i]*(-3.0*x12[i] + 234.0*x10[i]*y2[i] - 2145.0*x8[i]*y4[i] + 5148.0*x6[i]*y6[i] - 3861.0*x4[i]*y8[i] + 858.0*x2[i]*y10[i] - 39.0*y12[i] - 26.0*(-3.0*x2[i] - 3.0*y2[i] + 28.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]));
      prCofDY[totalAN*282+i] = 1.23186332318453*y[i]*(-x14[i] + 91.0*x12[i]*y2[i] - 1001.0*x10[i]*y4[i] + 3003.0*x8[i]*y6[i] - 3003.0*x6[i]*y8[i] + 1001.0*x4[i]*y10[i] - 91.0*x2[i]*y12[i] + y14[i] + 7.0*(x2[i] + y2[i] - 30.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]));
      prCofDY[totalAN*283+i] = -145.49552259692*x[i]*y[i]*z[i]*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      prCofDY[totalAN*284+i] = 13.7174494214084*y[i]*(-15.0*x14[i] + 455.0*x12[i]*y2[i] - 3003.0*x10[i]*y4[i] + 6435.0*x8[i]*y6[i] - 5005.0*x6[i]*y8[i] + 1365.0*x4[i]*y10[i] - 105.0*x2[i]*y12[i] + y14[i]);

      prCofDZ[totalAN*252+i] = 0;
      prCofDZ[totalAN*253+i] = 4.84985075323068*y[i]*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      prCofDZ[totalAN*254+i] = 73.9117993910719*x[i]*y[i]*z[i]*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      prCofDZ[totalAN*255+i] = -5.84324080093084*y[i]*(x2[i] + y2[i] - 28.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*256+i] = 81.0180057178837*x[i]*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 26.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      prCofDZ[totalAN*257+i] = 6.41931677269582*y[i]*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*258+i] = 31.4713874492004*x[i]*y[i]*z[i]*(261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      prCofDZ[totalAN*259+i] = 6.80404212865317*y[i]*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*260+i] = 105.571130411511*x[i]*y[i]*z[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i]);
      prCofDZ[totalAN*261+i] = 1.01013828545712*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]);
      prCofDZ[totalAN*262+i] = 9.76896009261221*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]);
      prCofDZ[totalAN*263+i] = 7.2530828229582*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]);
      prCofDZ[totalAN*264+i] = 39.8750200272244*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]);
      prCofDZ[totalAN*265+i] = 0.66975560245427*y[i]*(3.0*x2[i] - y2[i])*(190095.0*z12[i] - 432630.0*z10[i]*r2[i] + 360525.0*z8[i]*r4[i] - 134596.0*z6[i]*r6[i] + 21945.0*z4[i]*r8[i] - 1254.0*z2[i]*r10[i] + 11.0*r12[i]);
      prCofDZ[totalAN*266+i] = 0.530690686880522*x[i]*y[i]*z[i]*(570285.0*z12[i] - 1533870.0*z10[i]*r2[i] + 1562275.0*z8[i]*r4[i] - 749892.0*z6[i]*r6[i] + 171171.0*z4[i]*r8[i] - 16302.0*z2[i]*r10[i] + 429.0*r12[i]);
      prCofDZ[totalAN*267+i] = 0.0173018607299032*y[i]*(9694845.0*z14[i] - 30421755.0*z12[i]*r2[i] + 37182145.0*z10[i]*r4[i] - 22309287.0*z8[i]*r6[i] + 6789783.0*z6[i]*r8[i] - 969969.0*z4[i]*r10[i] + 51051.0*z2[i]*r12[i] - 429.0*r14[i]);
      prCofDZ[totalAN*268+i] = 0.0126602437781808*z[i]*(9694845.0*z14[i] - 35102025.0*z12[i]*r2[i] + 50702925.0*z10[i]*r4[i] - 37182145.0*z8[i]*r6[i] + 14549535.0*z6[i]*r8[i] - 2909907.0*z4[i]*r10[i] + 255255.0*z2[i]*r12[i] - 6435.0*r14[i]);
      prCofDZ[totalAN*269+i] = 0.0173018607299032*x[i]*(9694845.0*z14[i] - 30421755.0*z12[i]*r2[i] + 37182145.0*z10[i]*r4[i] - 22309287.0*z8[i]*r6[i] + 6789783.0*z6[i]*r8[i] - 969969.0*z4[i]*r10[i] + 51051.0*z2[i]*r12[i] - 429.0*r14[i]);
      prCofDZ[totalAN*270+i] = 0.265345343440261*z[i]*(x2[i] - y2[i])*(570285.0*z12[i] - 1533870.0*z10[i]*r2[i] + 1562275.0*z8[i]*r4[i] - 749892.0*z6[i]*r6[i] + 171171.0*z4[i]*r8[i] - 16302.0*z2[i]*r10[i] + 429.0*r12[i]);
      prCofDZ[totalAN*271+i] = 0.66975560245427*x[i]*(x2[i] - 3.0*y2[i])*(190095.0*z12[i] - 432630.0*z10[i]*r2[i] + 360525.0*z8[i]*r4[i] - 134596.0*z6[i]*r6[i] + 21945.0*z4[i]*r8[i] - 1254.0*z2[i]*r10[i] + 11.0*r12[i]);
      prCofDZ[totalAN*272+i] = 9.96875500680609*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(10005.0*z10[i] - 18975.0*z8[i]*r2[i] + 12650.0*z6[i]*r4[i] - 3542.0*z4[i]*r6[i] + 385.0*z2[i]*r8[i] - 11.0*r10[i]);
      prCofDZ[totalAN*273+i] = 7.2530828229582*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(10005.0*z10[i] - 15525.0*z8[i]*r2[i] + 8050.0*z6[i]*r4[i] - 1610.0*z4[i]*r6[i] + 105.0*z2[i]*r8[i] - r10[i]);
      prCofDZ[totalAN*274+i] = 4.88448004630611*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(10005.0*z8[i] - 12420.0*z6[i]*r2[i] + 4830.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 21.0*r8[i]);
      prCofDZ[totalAN*275+i] = 1.01013828545712*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(30015.0*z8[i] - 28980.0*z6[i]*r2[i] + 8050.0*z4[i]*r4[i] - 644.0*z2[i]*r6[i] + 7.0*r8[i]);
      prCofDZ[totalAN*276+i] = 13.1963913014389*z[i]*(1305.0*z6[i] - 945.0*z4[i]*r2[i] + 175.0*z2[i]*r4[i] - 7.0*r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*277+i] = 6.80404212865317*x[i]*(1305.0*z6[i] - 675.0*z4[i]*r2[i] + 75.0*z2[i]*r4[i] - r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      prCofDZ[totalAN*278+i] = 15.7356937246002*z[i]*(261.0*z4[i] - 90.0*z2[i]*r2[i] + 5.0*r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*279+i] = 6.41931677269582*x[i]*(261.0*z4[i] - 54.0*z2[i]*r2[i] + r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      prCofDZ[totalAN*280+i] = 20.2545014294709*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 26.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*281+i] = -5.84324080093084*x[i]*(x2[i] + y2[i] - 28.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      prCofDZ[totalAN*282+i] = 36.955899695536*z[i]*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      prCofDZ[totalAN*283+i] = 4.84985075323068*x[i]*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
      prCofDZ[totalAN*284+i] = 0;
        }
    if (lMax > 16){ 
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

        if(return_derivatives){
      prCofDX[totalAN*285+i] = 236.601150762411*x[i]*y[i]*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]);
      prCofDX[totalAN*286+i] = 81.1535251976858*y[i]*z[i]*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      prCofDX[totalAN*287+i] = 1.2486635518515*x[i]*y[i]*(-15.0*x14[i] + 455.0*x12[i]*y2[i] - 3003.0*x10[i]*y4[i] + 6435.0*x8[i]*y6[i] - 5005.0*x6[i]*y8[i] + 1365.0*x4[i]*y10[i] - 105.0*x2[i]*y12[i] + y14[i] - 15.0*(x2[i] + y2[i] - 32.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]));
      prCofDX[totalAN*288+i] = 12.2343542497898*y[i]*z[i]*(x2[i]*(-14.0*x12[i] + 364.0*x10[i]*y2[i] - 2002.0*x8[i]*y4[i] + 3432.0*x6[i]*y6[i] - 2002.0*x4[i]*y8[i] + 364.0*x2[i]*y10[i] - 14.0*y12[i]) + 7.0*(-x2[i] - y2[i] + 10.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]));
      prCofDX[totalAN*289+i] = 2.19735489013606*x[i]*y[i]*(-(-x2[i] - y2[i] + 30.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]) + 13.0*(341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]));
      prCofDX[totalAN*290+i] = 1.79413275488091*y[i]*z[i]*(-20.0*x2[i]*(-3.0*x2[i] - 3.0*y2[i] + 28.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]) + 3.0*(1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDX[totalAN*291+i] = 0.204019279709407*x[i]*y[i]*(-5.0*(899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 11.0*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDX[totalAN*292+i] = 0.408038559418814*y[i]*z[i]*(-14.0*x2[i]*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + 5.0*(9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*293+i] = 0.111054029550712*x[i]*y[i]*(9.0*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]) - 7.0*(8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*294+i] = 1.69879999122657*y[i]*z[i]*(-8.0*x2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i]) + (7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]));
      prCofDX[totalAN*295+i] = 0.134301931533751*x[i]*y[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]) - 5.0*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]));
      prCofDX[totalAN*296+i] = 0.727382699731321*y[i]*z[i]*(-10.0*x2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]) + 3.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]));
      prCofDX[totalAN*297+i] = 0.262699760480955*x[i]*y[i]*(5.0*(x2[i] - y2[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]) - (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]));
      prCofDX[totalAN*298+i] = 0.0310675249591503*y[i]*z[i]*(-52.0*x2[i]*(x2[i] - y2[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]) + (3.0*x2[i] - y2[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]));
      prCofDX[totalAN*299+i] = 0.00634163197675825*x[i]*y[i]*(30705345.0*z14[i] - 84672315.0*z12[i]*r2[i] + 90135045.0*z10[i]*r4[i] - 46621575.0*z8[i]*r6[i] + 12087075.0*z6[i]*r8[i] - 1450449.0*z4[i]*r10[i] + 63063.0*z2[i]*r12[i] - 91.0*(3.0*x2[i] - y2[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]) - 429.0*r14[i]);
      prCofDX[totalAN*300+i] = 0.0219680575732975*y[i]*z[i]*(-70.0*x2[i]*(310155.0*z12[i] - 780390.0*z10[i]*r2[i] + 740025.0*z8[i]*r4[i] - 328900.0*z6[i]*r6[i] + 69069.0*z4[i]*r8[i] - 6006.0*z2[i]*r10[i] + 143.0*r12[i]) + 3411705.0*z14[i] - 10855425.0*z12[i]*r2[i] + 13656825.0*z10[i]*r4[i] - 8633625.0*z8[i]*r6[i] + 2877875.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 35035.0*z2[i]*r12[i] - 715.0*r14[i]);
      prCofDX[totalAN*301+i] = -0.050398180498888*x[i]*y[i]*(5892945.0*z14[i] - 17298645.0*z12[i]*r2[i] + 19684665.0*z10[i]*r4[i] - 10935925.0*z8[i]*r6[i] + 3062059.0*z6[i]*r8[i] - 399399.0*z4[i]*r10[i] + 19019.0*z2[i]*r12[i] - 143.0*r14[i]);
      prCofDX[totalAN*302+i] = 0.0138531347690573*x[i]*z[i]*(-17678835.0*z14[i] + 59879925.0*z12[i]*r2[i] - 80528175.0*z10[i]*r4[i] + 54679625.0*z8[i]*r6[i] - 19684665.0*z6[i]*r8[i] + 3594591.0*z4[i]*r10[i] - 285285.0*z2[i]*r12[i] + 6435.0*r14[i]);
      prCofDX[totalAN*303+i] = -0.050398180498888*x2[i]*(5892945.0*z14[i] - 17298645.0*z12[i]*r2[i] + 19684665.0*z10[i]*r4[i] - 10935925.0*z8[i]*r6[i] + 3062059.0*z6[i]*r8[i] - 399399.0*z4[i]*r10[i] + 19019.0*z2[i]*r12[i] - 143.0*r14[i]) + 40836.6345447527*z16[i] - 148496.85289001*z14[i]*r2[i] + 217955.058274046*z12[i]*r4[i] - 165345.21662169*z10[i]*r6[i] + 68893.8402590377*z8[i]*r8[i] - 15432.2202180244*z6[i]*r10[i] + 1677.41524108961*z4[i]*r12[i] - 68.4659282077393*z2[i]*r14[i] + 0.450433738208811*r16[i];
      prCofDX[totalAN*304+i] = 0.0219680575732975*x[i]*z[i]*(3411705.0*z14[i] - 10855425.0*z12[i]*r2[i] + 13656825.0*z10[i]*r4[i] - 8633625.0*z8[i]*r6[i] + 2877875.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 35035.0*z2[i]*r12[i] - 35.0*(x2[i] - y2[i])*(310155.0*z12[i] - 780390.0*z10[i]*r2[i] + 740025.0*z8[i]*r4[i] - 328900.0*z6[i]*r6[i] + 69069.0*z4[i]*r8[i] - 6006.0*z2[i]*r10[i] + 143.0*r12[i]) - 715.0*r14[i]);
      prCofDX[totalAN*305+i] = -0.577088509885001*x2[i]*(x2[i] - 3.0*y2[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]) + 0.00951244796513738*(x2[i] - y2[i])*(10235115.0*z14[i] - 28224105.0*z12[i]*r2[i] + 30045015.0*z10[i]*r4[i] - 15540525.0*z8[i]*r6[i] + 4029025.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 21021.0*z2[i]*r12[i] - 143.0*r14[i]);
      prCofDX[totalAN*306+i] = 0.0310675249591503*x[i]*z[i]*((x2[i] - 3.0*y2[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]) - 13.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]));
      prCofDX[totalAN*307+i] = -0.262699760480955*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]) + 0.328374700601193*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]);
      prCofDX[totalAN*308+i] = 0.727382699731321*x[i]*z[i]*(3.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]) - 5.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]));
      prCofDX[totalAN*309+i] = -0.671509657668754*x2[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]) + 0.470056760368128*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]);
      prCofDX[totalAN*310+i] = 1.69879999122657*x[i]*z[i]*((x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]) - (8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*311+i] = -0.777378206854985*x2[i]*(8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) + 0.124935783244551*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]);
      prCofDX[totalAN*312+i] = 0.408038559418814*x[i]*z[i]*(-7.0*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) + 5.0*(9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDX[totalAN*313+i] = -1.02009639854704*x2[i]*(899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) + 1.12210603840174*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDX[totalAN*314+i] = 1.79413275488091*x[i]*z[i]*(-5.0*(-3.0*x2[i] - 3.0*y2[i] + 28.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]) + 3.0*(1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDX[totalAN*315+i] = -2.19735489013606*x2[i]*(-x2[i] - y2[i] + 30.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]) + 7.14140339294219*(341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDX[totalAN*316+i] = 12.2343542497898*x[i]*z[i]*(-x14[i] + 91.0*x12[i]*y2[i] - 1001.0*x10[i]*y4[i] + 3003.0*x8[i]*y6[i] - 3003.0*x6[i]*y8[i] + 1001.0*x4[i]*y10[i] - 91.0*x2[i]*y12[i] + y14[i] + 7.0*(-x2[i] - y2[i] + 10.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]));
      prCofDX[totalAN*317+i] = -1.2486635518515*x2[i]*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]) - 9.36497663888624*(x2[i] + y2[i] - 32.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      prCofDX[totalAN*318+i] = 81.1535251976858*x[i]*z[i]*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
      prCofDX[totalAN*319+i] = 14.7875719226507*x16[i] - 1774.50863071808*x14[i]*y2[i] + 26913.3808992242*x12[i]*y4[i] - 118418.875956587*x10[i]*y6[i] + 190316.050644514*x8[i]*y8[i] - 118418.875956587*x6[i]*y10[i] + 26913.3808992242*x4[i]*y12[i] - 1774.50863071808*x2[i]*y14[i] + 14.7875719226507*y16[i];

      prCofDY[totalAN*285+i] = 14.7875719226507*x16[i] - 1774.50863071808*x14[i]*y2[i] + 26913.3808992242*x12[i]*y4[i] - 118418.875956587*x10[i]*y6[i] + 190316.050644514*x8[i]*y8[i] - 118418.875956587*x6[i]*y10[i] + 26913.3808992242*x4[i]*y12[i] - 1774.50863071808*x2[i]*y14[i] + 14.7875719226507*y16[i];
      prCofDY[totalAN*286+i] = 81.1535251976858*x[i]*z[i]*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
      prCofDY[totalAN*287+i] = -1.2486635518515*y2[i]*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]) - 9.36497663888624*(x2[i] + y2[i] - 32.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      prCofDY[totalAN*288+i] = 12.2343542497898*x[i]*z[i]*(y2[i]*(-14.0*x12[i] + 364.0*x10[i]*y2[i] - 2002.0*x8[i]*y4[i] + 3432.0*x6[i]*y6[i] - 2002.0*x4[i]*y8[i] + 364.0*x2[i]*y10[i] - 14.0*y12[i]) + 7.0*(-x2[i] - y2[i] + 10.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]));
      prCofDY[totalAN*289+i] = -2.19735489013606*y2[i]*(-x2[i] - y2[i] + 30.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]) + 7.14140339294219*(341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDY[totalAN*290+i] = 1.79413275488091*x[i]*z[i]*(-20.0*y2[i]*(-3.0*x2[i] - 3.0*y2[i] + 28.0*z2[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]) + 3.0*(1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDY[totalAN*291+i] = -1.02009639854704*y2[i]*(899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 1.12210603840174*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDY[totalAN*292+i] = 0.408038559418814*x[i]*z[i]*(-14.0*y2[i]*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + 5.0*(9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDY[totalAN*293+i] = -0.777378206854985*y2[i]*(8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) + 0.124935783244551*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]);
      prCofDY[totalAN*294+i] = 1.69879999122657*x[i]*z[i]*(-8.0*y2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i]) + (x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]));
      prCofDY[totalAN*295+i] = -0.671509657668754*y2[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]) + 0.470056760368128*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]);
      prCofDY[totalAN*296+i] = 0.727382699731321*x[i]*z[i]*(-10.0*y2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]) + 3.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]));
      prCofDY[totalAN*297+i] = -0.262699760480955*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]) + 0.328374700601193*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]);
      prCofDY[totalAN*298+i] = 0.0310675249591503*x[i]*z[i]*(-52.0*y2[i]*(x2[i] - y2[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]) + (x2[i] - 3.0*y2[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]));
      prCofDY[totalAN*299+i] = -0.577088509885001*y2[i]*(3.0*x2[i] - y2[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]) + 0.00951244796513738*(x2[i] - y2[i])*(10235115.0*z14[i] - 28224105.0*z12[i]*r2[i] + 30045015.0*z10[i]*r4[i] - 15540525.0*z8[i]*r6[i] + 4029025.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 21021.0*z2[i]*r12[i] - 143.0*r14[i]);
      prCofDY[totalAN*300+i] = 0.0219680575732975*x[i]*z[i]*(-70.0*y2[i]*(310155.0*z12[i] - 780390.0*z10[i]*r2[i] + 740025.0*z8[i]*r4[i] - 328900.0*z6[i]*r6[i] + 69069.0*z4[i]*r8[i] - 6006.0*z2[i]*r10[i] + 143.0*r12[i]) + 3411705.0*z14[i] - 10855425.0*z12[i]*r2[i] + 13656825.0*z10[i]*r4[i] - 8633625.0*z8[i]*r6[i] + 2877875.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 35035.0*z2[i]*r12[i] - 715.0*r14[i]);
      prCofDY[totalAN*301+i] = -0.050398180498888*y2[i]*(5892945.0*z14[i] - 17298645.0*z12[i]*r2[i] + 19684665.0*z10[i]*r4[i] - 10935925.0*z8[i]*r6[i] + 3062059.0*z6[i]*r8[i] - 399399.0*z4[i]*r10[i] + 19019.0*z2[i]*r12[i] - 143.0*r14[i]) + 40836.6345447527*z16[i] - 148496.85289001*z14[i]*r2[i] + 217955.058274046*z12[i]*r4[i] - 165345.21662169*z10[i]*r6[i] + 68893.8402590377*z8[i]*r8[i] - 15432.2202180244*z6[i]*r10[i] + 1677.41524108961*z4[i]*r12[i] - 68.4659282077393*z2[i]*r14[i] + 0.450433738208811*r16[i];
      prCofDY[totalAN*302+i] = 0.0138531347690573*y[i]*z[i]*(-17678835.0*z14[i] + 59879925.0*z12[i]*r2[i] - 80528175.0*z10[i]*r4[i] + 54679625.0*z8[i]*r6[i] - 19684665.0*z6[i]*r8[i] + 3594591.0*z4[i]*r10[i] - 285285.0*z2[i]*r12[i] + 6435.0*r14[i]);
      prCofDY[totalAN*303+i] = -0.050398180498888*x[i]*y[i]*(5892945.0*z14[i] - 17298645.0*z12[i]*r2[i] + 19684665.0*z10[i]*r4[i] - 10935925.0*z8[i]*r6[i] + 3062059.0*z6[i]*r8[i] - 399399.0*z4[i]*r10[i] + 19019.0*z2[i]*r12[i] - 143.0*r14[i]);
      prCofDY[totalAN*304+i] = 0.0219680575732975*y[i]*z[i]*(-3411705.0*z14[i] + 10855425.0*z12[i]*r2[i] - 13656825.0*z10[i]*r4[i] + 8633625.0*z8[i]*r6[i] - 2877875.0*z6[i]*r8[i] + 483483.0*z4[i]*r10[i] - 35035.0*z2[i]*r12[i] - 35.0*(x2[i] - y2[i])*(310155.0*z12[i] - 780390.0*z10[i]*r2[i] + 740025.0*z8[i]*r4[i] - 328900.0*z6[i]*r6[i] + 69069.0*z4[i]*r8[i] - 6006.0*z2[i]*r10[i] + 143.0*r12[i]) + 715.0*r14[i]);
      prCofDY[totalAN*305+i] = 0.00634163197675825*x[i]*y[i]*(-30705345.0*z14[i] + 84672315.0*z12[i]*r2[i] - 90135045.0*z10[i]*r4[i] + 46621575.0*z8[i]*r6[i] - 12087075.0*z6[i]*r8[i] + 1450449.0*z4[i]*r10[i] - 63063.0*z2[i]*r12[i] - 91.0*(x2[i] - 3.0*y2[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]) + 429.0*r14[i]);
      prCofDY[totalAN*306+i] = -0.0310675249591503*y[i]*z[i]*((3.0*x2[i] - y2[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]) + 13.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]));
      prCofDY[totalAN*307+i] = -0.262699760480955*x[i]*y[i]*(5.0*(x2[i] - y2[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]) + (x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]));
      prCofDY[totalAN*308+i] = -0.727382699731321*y[i]*z[i]*(3.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]) + 5.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]));
      prCofDY[totalAN*309+i] = -0.134301931533751*x[i]*y[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]) + 5.0*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]));
      prCofDY[totalAN*310+i] = -1.69879999122657*y[i]*z[i]*((7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]) + (8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i])*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i]));
      prCofDY[totalAN*311+i] = -0.111054029550712*x[i]*y[i]*(9.0*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]) + 7.0*(8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDY[totalAN*312+i] = -0.408038559418814*y[i]*z[i]*(7.0*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) + 5.0*(9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDY[totalAN*313+i] = -0.204019279709407*x[i]*y[i]*(5.0*(899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) + 11.0*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDY[totalAN*314+i] = 1.79413275488091*y[i]*z[i]*(5.0*(3.0*x2[i] + 3.0*y2[i] - 28.0*z2[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]) - 3.0*(1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDY[totalAN*315+i] = 2.19735489013606*x[i]*y[i]*((x2[i] + y2[i] - 30.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]) - 13.0*(341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]));
      prCofDY[totalAN*316+i] = 12.2343542497898*y[i]*z[i]*(-x14[i] + 91.0*x12[i]*y2[i] - 1001.0*x10[i]*y4[i] + 3003.0*x8[i]*y6[i] - 3003.0*x6[i]*y8[i] + 1001.0*x4[i]*y10[i] - 91.0*x2[i]*y12[i] + y14[i] - 7.0*(-x2[i] - y2[i] + 10.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]));
      prCofDY[totalAN*317+i] = 1.2486635518515*x[i]*y[i]*(-x14[i] + 105.0*x12[i]*y2[i] - 1365.0*x10[i]*y4[i] + 5005.0*x8[i]*y6[i] - 6435.0*x6[i]*y8[i] + 3003.0*x4[i]*y10[i] - 455.0*x2[i]*y12[i] + 15.0*y14[i] + 15.0*(x2[i] + y2[i] - 32.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]));
      prCofDY[totalAN*318+i] = -81.1535251976858*y[i]*z[i]*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      prCofDY[totalAN*319+i] = 236.601150762411*x[i]*y[i]*(-x14[i] + 35.0*x12[i]*y2[i] - 273.0*x10[i]*y4[i] + 715.0*x8[i]*y6[i] - 715.0*x6[i]*y8[i] + 273.0*x4[i]*y10[i] - 35.0*x2[i]*y12[i] + y14[i]);

      prCofDZ[totalAN*285+i] = 0;
      prCofDZ[totalAN*286+i] = 81.1535251976858*x[i]*y[i]*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]);
      prCofDZ[totalAN*287+i] = 39.957233659248*y[i]*z[i]*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      prCofDZ[totalAN*288+i] = -12.2343542497898*x[i]*y[i]*(x2[i] + y2[i] - 30.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      prCofDZ[totalAN*289+i] = 21.9735489013606*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 28.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*290+i] = 8.97066377440456*x[i]*y[i]*(899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      prCofDZ[totalAN*291+i] = 5.7125398318634*y[i]*z[i]*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*292+i] = 2.8562699159317*x[i]*y[i]*(8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      prCofDZ[totalAN*293+i] = 2.88740476831852*y[i]*z[i]*(8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]);
      prCofDZ[totalAN*294+i] = 8.49399995613287*x[i]*y[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]);
      prCofDZ[totalAN*295+i] = 5.37207726135003*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]);
      prCofDZ[totalAN*296+i] = 0.727382699731321*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]);
      prCofDZ[totalAN*297+i] = 0.52539952096191*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]);
      prCofDZ[totalAN*298+i] = 2.82714477128268*x[i]*y[i]*(x2[i] - y2[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]);
      prCofDZ[totalAN*299+i] = 0.887828476746155*y[i]*z[i]*(3.0*x2[i] - y2[i])*(310155.0*z12[i] - 780390.0*z10[i]*r2[i] + 740025.0*z8[i]*r4[i] - 328900.0*z6[i]*r6[i] + 69069.0*z4[i]*r8[i] - 6006.0*z2[i]*r10[i] + 143.0*r12[i]);
      prCofDZ[totalAN*300+i] = 0.109840287866487*x[i]*y[i]*(5892945.0*z14[i] - 17298645.0*z12[i]*r2[i] + 19684665.0*z10[i]*r4[i] - 10935925.0*z8[i]*r6[i] + 3062059.0*z6[i]*r8[i] - 399399.0*z4[i]*r10[i] + 19019.0*z2[i]*r12[i] - 143.0*r14[i]);
      prCofDZ[totalAN*301+i] = 0.0201592721995552*y[i]*z[i]*(17678835.0*z14[i] - 59879925.0*z12[i]*r2[i] + 80528175.0*z10[i]*r4[i] - 54679625.0*z8[i]*r6[i] + 19684665.0*z6[i]*r8[i] - 3594591.0*z4[i]*r10[i] + 285285.0*z2[i]*r12[i] - 6435.0*r14[i]);
      prCofDZ[totalAN*302+i] = 260213.98905336*z16[i] - 1007279.95762591*z14[i]*r2[i] + 1580387.51972341*z12[i]*r4[i] - 1287723.16421907*z10[i]*r6[i] + 579475.423898583*z8[i]*r8[i] - 141089.668427481*z6[i]*r10[i] + 16796.3890985097*z4[i]*r12[i] - 757.731839030511*z2[i]*r14[i] + 5.57155763993023*r16[i];
      prCofDZ[totalAN*303+i] = 0.0201592721995552*x[i]*z[i]*(17678835.0*z14[i] - 59879925.0*z12[i]*r2[i] + 80528175.0*z10[i]*r4[i] - 54679625.0*z8[i]*r6[i] + 19684665.0*z6[i]*r8[i] - 3594591.0*z4[i]*r10[i] + 285285.0*z2[i]*r12[i] - 6435.0*r14[i]);
      prCofDZ[totalAN*304+i] = 0.0549201439332437*(x2[i] - y2[i])*(5892945.0*z14[i] - 17298645.0*z12[i]*r2[i] + 19684665.0*z10[i]*r4[i] - 10935925.0*z8[i]*r6[i] + 3062059.0*z6[i]*r8[i] - 399399.0*z4[i]*r10[i] + 19019.0*z2[i]*r12[i] - 143.0*r14[i]);
      prCofDZ[totalAN*305+i] = 0.887828476746155*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(310155.0*z12[i] - 780390.0*z10[i]*r2[i] + 740025.0*z8[i]*r4[i] - 328900.0*z6[i]*r6[i] + 69069.0*z4[i]*r8[i] - 6006.0*z2[i]*r10[i] + 143.0*r12[i]);
      prCofDZ[totalAN*306+i] = 0.706786192820669*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(310155.0*z12[i] - 660330.0*z10[i]*r2[i] + 512325.0*z8[i]*r4[i] - 177100.0*z6[i]*r6[i] + 26565.0*z4[i]*r8[i] - 1386.0*z2[i]*r10[i] + 11.0*r12[i]);
      prCofDZ[totalAN*307+i] = 0.52539952096191*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(310155.0*z10[i] - 550275.0*z8[i]*r2[i] + 341550.0*z6[i]*r4[i] - 88550.0*z4[i]*r6[i] + 8855.0*z2[i]*r8[i] - 231.0*r10[i]);
      prCofDZ[totalAN*308+i] = 0.36369134986566*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(310155.0*z10[i] - 450225.0*z8[i]*r2[i] + 217350.0*z6[i]*r4[i] - 40250.0*z4[i]*r6[i] + 2415.0*z2[i]*r8[i] - 21.0*r10[i]);
      prCofDZ[totalAN*309+i] = 5.37207726135003*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(13485.0*z8[i] - 15660.0*z6[i]*r2[i] + 5670.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 21.0*r8[i]);
      prCofDZ[totalAN*310+i] = 1.06174999451661*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(40455.0*z8[i] - 36540.0*z6[i]*r2[i] + 9450.0*z4[i]*r4[i] - 700.0*z2[i]*r6[i] + 7.0*r8[i]);
      prCofDZ[totalAN*311+i] = 2.88740476831852*x[i]*z[i]*(8091.0*z6[i] - 5481.0*z4[i]*r2[i] + 945.0*z2[i]*r4[i] - 35.0*r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]);
      prCofDZ[totalAN*312+i] = 1.42813495796585*(8091.0*z6[i] - 3915.0*z4[i]*r2[i] + 405.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*313+i] = 5.7125398318634*x[i]*z[i]*(899.0*z4[i] - 290.0*z2[i]*r2[i] + 15.0*r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      prCofDZ[totalAN*314+i] = 2.24266594360114*(899.0*z4[i] - 174.0*z2[i]*r2[i] + 3.0*r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*315+i] = 21.9735489013606*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 28.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      prCofDZ[totalAN*316+i] = -6.11717712489491*(x2[i] + y2[i] - 30.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      prCofDZ[totalAN*317+i] = 39.957233659248*x[i]*z[i]*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
      prCofDZ[totalAN*318+i] = 5.07209532485536*x16[i] - 608.651438982643*x14[i]*y2[i] + 9231.21349123676*x12[i]*y4[i] - 40617.3393614417*x10[i]*y6[i] + 65277.8668308885*x8[i]*y8[i] - 40617.3393614417*x6[i]*y10[i] + 9231.21349123676*x4[i]*y12[i] - 608.651438982643*x2[i]*y14[i] + 5.07209532485536*y16[i];
      prCofDZ[totalAN*319+i] = 0;
        }
    if (lMax > 17){ 
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

        if(return_derivatives){
      prCofDX[totalAN*320+i] = 15.8734038362099*y[i]*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]);
      prCofDX[totalAN*321+i] = 1439.18861448303*x[i]*y[i]*z[i]*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]);
      prCofDX[totalAN*322+i] = -10.1185847426968*y[i]*(2.0*x2[i]*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]) + (x2[i] + y2[i] - 34.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]));
      prCofDX[totalAN*323+i] = 12.7740870722626*x[i]*y[i]*z[i]*(-15.0*x14[i] + 455.0*x12[i]*y2[i] - 3003.0*x10[i]*y4[i] + 6435.0*x8[i]*y6[i] - 5005.0*x6[i]*y8[i] + 1365.0*x4[i]*y10[i] - 105.0*x2[i]*y12[i] + y14[i] + 5.0*(-3.0*x2[i] - 3.0*y2[i] + 32.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]));
      prCofDX[totalAN*324+i] = 1.11184156725673*y[i]*(-4.0*x2[i]*(-x2[i] - y2[i] + 32.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]) + 7.0*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]));
      prCofDX[totalAN*325+i] = 28.1276139982605*x[i]*y[i]*z[i]*(-(-x2[i] - y2[i] + 10.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]) + 13.0*(77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]));
      prCofDX[totalAN*326+i] = 6.18725016791869*y[i]*(-2.0*x2[i]*(341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]) + (2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDX[totalAN*327+i] = 2.98872577354112*x[i]*y[i]*z[i]*(-(1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 11.0*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDX[totalAN*328+i] = 0.1962194600598*y[i]*(-8.0*x2[i]*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + 5.0*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDX[totalAN*329+i] = 1.77993563557578*x[i]*y[i]*z[i]*((x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]) - (9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]));
      prCofDX[totalAN*330+i] = 0.541617165840082*y[i]*(-2.0*x2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]) + (7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]));
      prCofDX[totalAN*331+i] = 2.28989434989826*x[i]*y[i]*z[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]) - (7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]));
      prCofDX[totalAN*332+i] = 0.39662133579887*y[i]*(-4.0*x2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]) + (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]));
      prCofDX[totalAN*333+i] = 0.35926818382145*x[i]*y[i]*z[i]*(5.0*(x2[i] - y2[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]) - 13.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]));
      prCofDX[totalAN*334+i] = 0.140148631920647*y[i]*(-26.0*x2[i]*(x2[i] - y2[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]) + (3.0*x2[i] - y2[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]));
      prCofDX[totalAN*335+i] = 0.115723924107498*x[i]*y[i]*z[i]*(3411705.0*z14[i] - 10235115.0*z12[i]*r2[i] + 12096045.0*z10[i]*r4[i] - 7153575.0*z8[i]*r6[i] + 2220075.0*z6[i]*r8[i] - 345345.0*z4[i]*r10[i] + 23023.0*z2[i]*r12[i] - (3.0*x2[i] - y2[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]) - 429.0*r14[i]);
      prCofDX[totalAN*336+i] = 0.0063132576421421*y[i]*(-16.0*x2[i]*(10235115.0*z14[i] - 28224105.0*z12[i]*r2[i] + 30045015.0*z10[i]*r4[i] - 15540525.0*z8[i]*r6[i] + 4029025.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 21021.0*z2[i]*r12[i] - 143.0*r14[i]) + 23881935.0*z16[i] - 81880920.0*z14[i]*r2[i] + 112896420.0*z12[i]*r4[i] - 80120040.0*z10[i]*r6[i] + 31081050.0*z8[i]*r8[i] - 6446440.0*z6[i]*r10[i] + 644644.0*z4[i]*r12[i] - 24024.0*z2[i]*r14[i] + 143.0*r16[i]);
      prCofDX[totalAN*337+i] = -0.186257150406634*x[i]*y[i]*z[i]*(3411705.0*z14[i] - 10855425.0*z12[i]*r2[i] + 13656825.0*z10[i]*r4[i] - 8633625.0*z8[i]*r6[i] + 2877875.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 35035.0*z2[i]*r12[i] - 715.0*r14[i]);
      prCofDX[totalAN*338+i] = 0.00801193515922079*x[i]*(-64822395.0*z16[i] + 235717800.0*z14[i]*r2[i] - 345972900.0*z12[i]*r4[i] + 262462200.0*z10[i]*r6[i] - 109359250.0*z8[i]*r8[i] + 24496472.0*z6[i]*r10[i] - 2662660.0*z4[i]*r12[i] + 108680.0*z2[i]*r14[i] - 715.0*r16[i]);
      prCofDX[totalAN*339+i] = 0.000684768935318508*z[i]*(-272.0*x2[i]*(3411705.0*z14[i] - 10855425.0*z12[i]*r2[i] + 13656825.0*z10[i]*r4[i] - 8633625.0*z8[i]*r6[i] + 2877875.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 35035.0*z2[i]*r12[i] - 715.0*r14[i]) + 119409675.0*z16[i] - 463991880.0*z14[i]*r2[i] + 738168900.0*z12[i]*r4[i] - 619109400.0*z10[i]*r6[i] + 293543250.0*z8[i]*r8[i] - 78278200.0*z6[i]*r10[i] + 10958948.0*z4[i]*r12[i] - 680680.0*z2[i]*r14[i] + 12155.0*r16[i]);
      prCofDX[totalAN*340+i] = 0.0063132576421421*x[i]*(23881935.0*z16[i] - 81880920.0*z14[i]*r2[i] + 112896420.0*z12[i]*r4[i] - 80120040.0*z10[i]*r6[i] + 31081050.0*z8[i]*r8[i] - 6446440.0*z6[i]*r10[i] + 644644.0*z4[i]*r12[i] - 24024.0*z2[i]*r14[i] - 8.0*(x2[i] - y2[i])*(10235115.0*z14[i] - 28224105.0*z12[i]*r2[i] + 30045015.0*z10[i]*r4[i] - 15540525.0*z8[i]*r6[i] + 4029025.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 21021.0*z2[i]*r12[i] - 143.0*r14[i]) + 143.0*r16[i]);
      prCofDX[totalAN*341+i] = 0.0578619620537492*z[i]*(-2.0*x2[i]*(x2[i] - 3.0*y2[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]) + (x2[i] - y2[i])*(3411705.0*z14[i] - 10235115.0*z12[i]*r2[i] + 12096045.0*z10[i]*r4[i] - 7153575.0*z8[i]*r6[i] + 2220075.0*z6[i]*r8[i] - 345345.0*z4[i]*r10[i] + 23023.0*z2[i]*r12[i] - 429.0*r14[i]));
      prCofDX[totalAN*342+i] = 0.0700743159603237*x[i]*(2.0*(x2[i] - 3.0*y2[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]) - 13.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]));
      prCofDX[totalAN*343+i] = 0.0898170459553624*z[i]*(-52.0*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]) + 5.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]));
      prCofDX[totalAN*344+i] = 0.39662133579887*x[i]*((x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]) - 2.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]));
      prCofDX[totalAN*345+i] = 1.14494717494913*z[i]*(-2.0*x2[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]) + 7.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]));
      prCofDX[totalAN*346+i] = 0.135404291460021*x[i]*(4.0*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]) - (x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]));
      prCofDX[totalAN*347+i] = 0.222491954446972*z[i]*(-8.0*x2[i]*(9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]) + (x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]));
      prCofDX[totalAN*348+i] = 0.1962194600598*x[i]*(-4.0*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) + 5.0*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDX[totalAN*349+i] = 1.49436288677056*z[i]*(-2.0*x2[i]*(1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) + 11.0*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]));
      prCofDX[totalAN*350+i] = 3.09362508395935*x[i]*(-(341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]) + 2.0*(2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDX[totalAN*351+i] = 7.03190349956512*z[i]*(-4.0*x2[i]*(-x2[i] - y2[i] + 10.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]) + 13.0*(77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]));
      prCofDX[totalAN*352+i] = 1.11184156725673*x[i]*(-2.0*(-x2[i] - y2[i] + 32.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]) + 7.0*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]));
      prCofDX[totalAN*353+i] = 6.38704353613131*z[i]*(-2.0*x2[i]*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]) + 5.0*(-3.0*x2[i] - 3.0*y2[i] + 32.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]));
      prCofDX[totalAN*354+i] = 1.26482309283709*x[i]*(-x16[i] + 120.0*x14[i]*y2[i] - 1820.0*x12[i]*y4[i] + 8008.0*x10[i]*y6[i] - 12870.0*x8[i]*y8[i] + 8008.0*x6[i]*y10[i] - 1820.0*x4[i]*y12[i] + 120.0*x2[i]*y14[i] - y16[i] - 8.0*(x2[i] + y2[i] - 34.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]));
      prCofDX[totalAN*355+i] = 89.9492884051895*z[i]*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]);
      prCofDX[totalAN*356+i] = 15.8734038362099*x[i]*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]);

      prCofDY[totalAN*320+i] = 15.8734038362099*x[i]*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]);
      prCofDY[totalAN*321+i] = 89.9492884051895*z[i]*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]);
      prCofDY[totalAN*322+i] = -10.1185847426968*x[i]*(2.0*y2[i]*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]) + (x2[i] + y2[i] - 34.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]));
      prCofDY[totalAN*323+i] = 6.38704353613131*z[i]*(-2.0*y2[i]*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]) + 5.0*(-3.0*x2[i] - 3.0*y2[i] + 32.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]));
      prCofDY[totalAN*324+i] = 1.11184156725673*x[i]*(-4.0*y2[i]*(-x2[i] - y2[i] + 32.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]) + 7.0*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]));
      prCofDY[totalAN*325+i] = 7.03190349956512*z[i]*(-4.0*y2[i]*(-x2[i] - y2[i] + 10.0*z2[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]) + 13.0*(77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]));
      prCofDY[totalAN*326+i] = 6.18725016791869*x[i]*(-2.0*y2[i]*(341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]) + (2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDY[totalAN*327+i] = 1.49436288677056*z[i]*(-2.0*y2[i]*(1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 11.0*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]));
      prCofDY[totalAN*328+i] = 0.1962194600598*x[i]*(-8.0*y2[i]*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + 5.0*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDY[totalAN*329+i] = 0.222491954446972*z[i]*(-8.0*y2[i]*(9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i]) + (x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]));
      prCofDY[totalAN*330+i] = 0.541617165840082*x[i]*(-2.0*y2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]) + (x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]));
      prCofDY[totalAN*331+i] = 1.14494717494913*z[i]*(-2.0*y2[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]) + 7.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]));
      prCofDY[totalAN*332+i] = 0.39662133579887*x[i]*(-4.0*y2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]) + (x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]));
      prCofDY[totalAN*333+i] = 0.0898170459553624*z[i]*(-52.0*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]) + 5.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]));
      prCofDY[totalAN*334+i] = 0.140148631920647*x[i]*(-26.0*y2[i]*(x2[i] - y2[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]) + (x2[i] - 3.0*y2[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]));
      prCofDY[totalAN*335+i] = 0.0578619620537492*z[i]*(-2.0*y2[i]*(3.0*x2[i] - y2[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]) + (x2[i] - y2[i])*(3411705.0*z14[i] - 10235115.0*z12[i]*r2[i] + 12096045.0*z10[i]*r4[i] - 7153575.0*z8[i]*r6[i] + 2220075.0*z6[i]*r8[i] - 345345.0*z4[i]*r10[i] + 23023.0*z2[i]*r12[i] - 429.0*r14[i]));
      prCofDY[totalAN*336+i] = 0.0063132576421421*x[i]*(-16.0*y2[i]*(10235115.0*z14[i] - 28224105.0*z12[i]*r2[i] + 30045015.0*z10[i]*r4[i] - 15540525.0*z8[i]*r6[i] + 4029025.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 21021.0*z2[i]*r12[i] - 143.0*r14[i]) + 23881935.0*z16[i] - 81880920.0*z14[i]*r2[i] + 112896420.0*z12[i]*r4[i] - 80120040.0*z10[i]*r6[i] + 31081050.0*z8[i]*r8[i] - 6446440.0*z6[i]*r10[i] + 644644.0*z4[i]*r12[i] - 24024.0*z2[i]*r14[i] + 143.0*r16[i]);
      prCofDY[totalAN*337+i] = 0.000684768935318508*z[i]*(-272.0*y2[i]*(3411705.0*z14[i] - 10855425.0*z12[i]*r2[i] + 13656825.0*z10[i]*r4[i] - 8633625.0*z8[i]*r6[i] + 2877875.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 35035.0*z2[i]*r12[i] - 715.0*r14[i]) + 119409675.0*z16[i] - 463991880.0*z14[i]*r2[i] + 738168900.0*z12[i]*r4[i] - 619109400.0*z10[i]*r6[i] + 293543250.0*z8[i]*r8[i] - 78278200.0*z6[i]*r10[i] + 10958948.0*z4[i]*r12[i] - 680680.0*z2[i]*r14[i] + 12155.0*r16[i]);
      prCofDY[totalAN*338+i] = 0.00801193515922079*y[i]*(-64822395.0*z16[i] + 235717800.0*z14[i]*r2[i] - 345972900.0*z12[i]*r4[i] + 262462200.0*z10[i]*r6[i] - 109359250.0*z8[i]*r8[i] + 24496472.0*z6[i]*r10[i] - 2662660.0*z4[i]*r12[i] + 108680.0*z2[i]*r14[i] - 715.0*r16[i]);
      prCofDY[totalAN*339+i] = -0.186257150406634*x[i]*y[i]*z[i]*(3411705.0*z14[i] - 10855425.0*z12[i]*r2[i] + 13656825.0*z10[i]*r4[i] - 8633625.0*z8[i]*r6[i] + 2877875.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 35035.0*z2[i]*r12[i] - 715.0*r14[i]);
      prCofDY[totalAN*340+i] = 0.0063132576421421*y[i]*(-23881935.0*z16[i] + 81880920.0*z14[i]*r2[i] - 112896420.0*z12[i]*r4[i] + 80120040.0*z10[i]*r6[i] - 31081050.0*z8[i]*r8[i] + 6446440.0*z6[i]*r10[i] - 644644.0*z4[i]*r12[i] + 24024.0*z2[i]*r14[i] - 8.0*(x2[i] - y2[i])*(10235115.0*z14[i] - 28224105.0*z12[i]*r2[i] + 30045015.0*z10[i]*r4[i] - 15540525.0*z8[i]*r6[i] + 4029025.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 21021.0*z2[i]*r12[i] - 143.0*r14[i]) - 143.0*r16[i]);
      prCofDY[totalAN*341+i] = 0.115723924107498*x[i]*y[i]*z[i]*(-3411705.0*z14[i] + 10235115.0*z12[i]*r2[i] - 12096045.0*z10[i]*r4[i] + 7153575.0*z8[i]*r6[i] - 2220075.0*z6[i]*r8[i] + 345345.0*z4[i]*r10[i] - 23023.0*z2[i]*r12[i] - (x2[i] - 3.0*y2[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]) + 429.0*r14[i]);
      prCofDY[totalAN*342+i] = -0.0700743159603237*y[i]*(2.0*(3.0*x2[i] - y2[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]) + 13.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]));
      prCofDY[totalAN*343+i] = -0.35926818382145*x[i]*y[i]*z[i]*(5.0*(x2[i] - y2[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]) + 13.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]));
      prCofDY[totalAN*344+i] = -0.39662133579887*y[i]*((5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]) + 2.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]));
      prCofDY[totalAN*345+i] = -2.28989434989826*x[i]*y[i]*z[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]) + (x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]));
      prCofDY[totalAN*346+i] = -0.135404291460021*y[i]*(4.0*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]) + (x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]));
      prCofDY[totalAN*347+i] = -1.77993563557578*x[i]*y[i]*z[i]*((x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]) + (9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i]));
      prCofDY[totalAN*348+i] = -0.1962194600598*y[i]*(4.0*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) + 5.0*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDY[totalAN*349+i] = -2.98872577354112*x[i]*y[i]*z[i]*((1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) + 11.0*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]));
      prCofDY[totalAN*350+i] = -3.09362508395935*y[i]*((341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]) + 2.0*(2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDY[totalAN*351+i] = 28.1276139982605*x[i]*y[i]*z[i]*((x2[i] + y2[i] - 10.0*z2[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]) - 13.0*(77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]));
      prCofDY[totalAN*352+i] = 1.11184156725673*y[i]*(2.0*(x2[i] + y2[i] - 32.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]) - 7.0*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]));
      prCofDY[totalAN*353+i] = 12.7740870722626*x[i]*y[i]*z[i]*(-x14[i] + 105.0*x12[i]*y2[i] - 1365.0*x10[i]*y4[i] + 5005.0*x8[i]*y6[i] - 6435.0*x6[i]*y8[i] + 3003.0*x4[i]*y10[i] - 455.0*x2[i]*y12[i] + 15.0*y14[i] - 5.0*(-3.0*x2[i] - 3.0*y2[i] + 32.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]));
      prCofDY[totalAN*354+i] = 1.26482309283709*y[i]*(-x16[i] + 120.0*x14[i]*y2[i] - 1820.0*x12[i]*y4[i] + 8008.0*x10[i]*y6[i] - 12870.0*x8[i]*y8[i] + 8008.0*x6[i]*y10[i] - 1820.0*x4[i]*y12[i] + 120.0*x2[i]*y14[i] - y16[i] + 8.0*(x2[i] + y2[i] - 34.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]));
      prCofDY[totalAN*355+i] = -1439.18861448303*x[i]*y[i]*z[i]*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]);
      prCofDY[totalAN*356+i] = 15.8734038362099*y[i]*(-17.0*x16[i] + 680.0*x14[i]*y2[i] - 6188.0*x12[i]*y4[i] + 19448.0*x10[i]*y6[i] - 24310.0*x8[i]*y8[i] + 12376.0*x6[i]*y10[i] - 2380.0*x4[i]*y12[i] + 136.0*x2[i]*y14[i] - y16[i]);

      prCofDZ[totalAN*320+i] = 0;
      prCofDZ[totalAN*321+i] = 5.29113461206997*y[i]*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]);
      prCofDZ[totalAN*322+i] = 688.063762503379*x[i]*y[i]*z[i]*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]);
      prCofDZ[totalAN*323+i] = -6.38704353613131*y[i]*(x2[i] + y2[i] - 32.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      prCofDZ[totalAN*324+i] = 142.315720608862*x[i]*y[i]*z[i]*(-x2[i] - y2[i] + 10.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      prCofDZ[totalAN*325+i] = 7.03190349956512*y[i]*(341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*326+i] = 24.7490006716748*x[i]*y[i]*z[i]*(1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      prCofDZ[totalAN*327+i] = 1.49436288677056*y[i]*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*328+i] = 6.2790227219136*x[i]*y[i]*z[i]*(9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]);
      prCofDZ[totalAN*329+i] = 0.222491954446972*y[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]);
      prCofDZ[totalAN*330+i] = 28.1640926236843*x[i]*y[i]*z[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]);
      prCofDZ[totalAN*331+i] = 1.14494717494913*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]);
      prCofDZ[totalAN*332+i] = 12.6918827455638*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]);
      prCofDZ[totalAN*333+i] = 1.16762159741971*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]);
      prCofDZ[totalAN*334+i] = 0.56059452768259*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]);
      prCofDZ[totalAN*335+i] = 0.0578619620537492*y[i]*(3.0*x2[i] - y2[i])*(10235115.0*z14[i] - 28224105.0*z12[i]*r2[i] + 30045015.0*z10[i]*r4[i] - 15540525.0*z8[i]*r6[i] + 4029025.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 21021.0*z2[i]*r12[i] - 143.0*r14[i]);
      prCofDZ[totalAN*336+i] = 0.404048489097094*x[i]*y[i]*z[i]*(3411705.0*z14[i] - 10855425.0*z12[i]*r2[i] + 13656825.0*z10[i]*r4[i] - 8633625.0*z8[i]*r6[i] + 2877875.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 35035.0*z2[i]*r12[i] - 715.0*r14[i]);
      prCofDZ[totalAN*337+i] = 0.0116410719004146*y[i]*(64822395.0*z16[i] - 235717800.0*z14[i]*r2[i] + 345972900.0*z12[i]*r4[i] - 262462200.0*z10[i]*r6[i] + 109359250.0*z8[i]*r8[i] - 24496472.0*z6[i]*r10[i] + 2662660.0*z4[i]*r12[i] - 108680.0*z2[i]*r14[i] + 715.0*r16[i]);
      prCofDZ[totalAN*338+i] = 0.000942580606967152*z[i]*(583401555.0*z16[i] - 2404321560.0*z14[i]*r2[i] + 4071834900.0*z12[i]*r4[i] - 3650610600.0*z10[i]*r6[i] + 1859107250.0*z8[i]*r8[i] - 535422888.0*z6[i]*r10[i] + 81477396.0*z4[i]*r12[i] - 5542680.0*z2[i]*r14[i] + 109395.0*r16[i]);
      prCofDZ[totalAN*339+i] = 0.0116410719004146*x[i]*(64822395.0*z16[i] - 235717800.0*z14[i]*r2[i] + 345972900.0*z12[i]*r4[i] - 262462200.0*z10[i]*r6[i] + 109359250.0*z8[i]*r8[i] - 24496472.0*z6[i]*r10[i] + 2662660.0*z4[i]*r12[i] - 108680.0*z2[i]*r14[i] + 715.0*r16[i]);
      prCofDZ[totalAN*340+i] = 0.202024244548547*z[i]*(x2[i] - y2[i])*(3411705.0*z14[i] - 10855425.0*z12[i]*r2[i] + 13656825.0*z10[i]*r4[i] - 8633625.0*z8[i]*r6[i] + 2877875.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 35035.0*z2[i]*r12[i] - 715.0*r14[i]);
      prCofDZ[totalAN*341+i] = 0.0578619620537492*x[i]*(x2[i] - 3.0*y2[i])*(10235115.0*z14[i] - 28224105.0*z12[i]*r2[i] + 30045015.0*z10[i]*r4[i] - 15540525.0*z8[i]*r6[i] + 4029025.0*z6[i]*r8[i] - 483483.0*z4[i]*r10[i] + 21021.0*z2[i]*r12[i] - 143.0*r14[i]);
      prCofDZ[totalAN*342+i] = 0.140148631920647*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(3411705.0*z12[i] - 8064030.0*z10[i]*r2[i] + 7153575.0*z8[i]*r4[i] - 2960100.0*z6[i]*r6[i] + 575575.0*z4[i]*r8[i] - 46046.0*z2[i]*r10[i] + 1001.0*r12[i]);
      prCofDZ[totalAN*343+i] = 1.16762159741971*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(310155.0*z12[i] - 620310.0*z10[i]*r2[i] + 450225.0*z8[i]*r4[i] - 144900.0*z6[i]*r6[i] + 20125.0*z4[i]*r8[i] - 966.0*z2[i]*r10[i] + 7.0*r12[i]);
      prCofDZ[totalAN*344+i] = 6.34594137278192*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(40455.0*z10[i] - 67425.0*z8[i]*r2[i] + 39150.0*z6[i]*r4[i] - 9450.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 21.0*r10[i]);
      prCofDZ[totalAN*345+i] = 1.14494717494913*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(148335.0*z10[i] - 202275.0*z8[i]*r2[i] + 91350.0*z6[i]*r4[i] - 15750.0*z4[i]*r6[i] + 875.0*z2[i]*r8[i] - 7.0*r10[i]);
      prCofDZ[totalAN*346+i] = 3.52051157796053*z[i]*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(29667.0*z8[i] - 32364.0*z6[i]*r2[i] + 10962.0*z4[i]*r4[i] - 1260.0*z2[i]*r6[i] + 35.0*r8[i]);
      prCofDZ[totalAN*347+i] = 0.222491954446972*x[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(267003.0*z8[i] - 226548.0*z6[i]*r2[i] + 54810.0*z4[i]*r4[i] - 3780.0*z2[i]*r6[i] + 35.0*r8[i]);
      prCofDZ[totalAN*348+i] = 3.1395113609568*z[i]*(9889.0*z6[i] - 6293.0*z4[i]*r2[i] + 1015.0*z2[i]*r4[i] - 35.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*349+i] = 1.49436288677056*x[i]*(9889.0*z6[i] - 4495.0*z4[i]*r2[i] + 435.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      prCofDZ[totalAN*350+i] = 6.18725016791869*z[i]*(1023.0*z4[i] - 310.0*z2[i]*r2[i] + 15.0*r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*351+i] = 7.03190349956512*x[i]*(341.0*z4[i] - 62.0*z2[i]*r2[i] + r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      prCofDZ[totalAN*352+i] = 71.1578603044308*z[i]*(-x2[i] - y2[i] + 10.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      prCofDZ[totalAN*353+i] = -6.38704353613131*x[i]*(x2[i] + y2[i] - 32.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
      prCofDZ[totalAN*354+i] = 43.0039851564612*z[i]*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]);
      prCofDZ[totalAN*355+i] = 5.29113461206997*x[i]*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]);
      prCofDZ[totalAN*356+i] = 0;
        }
    if (lMax > 18){ 
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

        if(return_derivatives){
      prCofDX[totalAN*357+i] = 33.9485838052981*x[i]*y[i]*(9.0*x16[i] - 408.0*x14[i]*y2[i] + 4284.0*x12[i]*y4[i] - 15912.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 15912.0*x6[i]*y10[i] + 4284.0*x4[i]*y12[i] - 408.0*x2[i]*y14[i] + 9.0*y16[i]);
      prCofDX[totalAN*358+i] = 99.1293751849004*y[i]*z[i]*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]);
      prCofDX[totalAN*359+i] = 1.2803950883772*x[i]*y[i]*(-17.0*x16[i] + 680.0*x14[i]*y2[i] - 6188.0*x12[i]*y4[i] + 19448.0*x10[i]*y6[i] - 24310.0*x8[i]*y8[i] + 12376.0*x6[i]*y10[i] - 2380.0*x4[i]*y12[i] + 136.0*x2[i]*y14[i] - y16[i] - 136.0*(x2[i] + y2[i] - 36.0*z2[i])*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]));
      prCofDX[totalAN*360+i] = 35.4833495492953*y[i]*z[i]*(-6.0*x2[i]*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]) + (-3.0*x2[i] - 3.0*y2[i] + 34.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]));
      prCofDX[totalAN*361+i] = 1.12458389413523*x[i]*y[i]*(-2.0*(-x2[i] - y2[i] + 34.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]) + 5.0*(1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]));
      prCofDX[totalAN*362+i] = 4.8875933516657*y[i]*z[i]*(-4.0*x2[i]*(-3.0*x2[i] - 3.0*y2[i] + 32.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]) + 7.0*(259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]));
      prCofDX[totalAN*363+i] = 1.04203840383005*x[i]*y[i]*(-3.0*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]) + 26.0*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]));
      prCofDX[totalAN*364+i] = 93.5748165838277*y[i]*z[i]*(-2.0*x2[i]*(77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]) + (407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDX[totalAN*365+i] = 0.990334465847138*x[i]*y[i]*(-4.0*(2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 11.0*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i])*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i]));
      prCofDX[totalAN*366+i] = 0.361619017612871*y[i]*z[i]*(-72.0*x2[i]*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + 5.0*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i]));
      prCofDX[totalAN*367+i] = 0.955574995064948*x[i]*y[i]*(4.0*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i]) - (9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDX[totalAN*368+i] = 1.06477924459395*y[i]*z[i]*(-2.0*x2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]) + (7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i]));
      prCofDX[totalAN*369+i] = 0.133097405574243*x[i]*y[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i]) - 6.0*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]));
      prCofDX[totalAN*370+i] = 0.564684468239301*y[i]*z[i]*(-52.0*x2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]) + (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i]));
      prCofDX[totalAN*371+i] = 0.0704285270594699*x[i]*y[i]*(10.0*(x2[i] - y2[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i]) - 13.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]));
      prCofDX[totalAN*372+i] = 0.890858231034903*y[i]*z[i]*(-6.0*x2[i]*(x2[i] - y2[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]) + (3.0*x2[i] - y2[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]));
      prCofDX[totalAN*373+i] = 0.0696587932695619*x[i]*y[i]*(11475735.0*z16[i] - 37218600.0*z14[i]*r2[i] + 48384180.0*z12[i]*r4[i] - 32256120.0*z10[i]*r6[i] + 11705850.0*z8[i]*r8[i] - 2260440.0*z6[i]*r10[i] + 209300.0*z4[i]*r12[i] - 7176.0*z2[i]*r14[i] - 8.0*(3.0*x2[i] - y2[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]) + 39.0*r16[i]);
      prCofDX[totalAN*374+i] = 0.00240131363330655*y[i]*z[i]*(-272.0*x2[i]*(3411705.0*z14[i] - 10235115.0*z12[i]*r2[i] + 12096045.0*z10[i]*r4[i] - 7153575.0*z8[i]*r6[i] + 2220075.0*z6[i]*r8[i] - 345345.0*z4[i]*r10[i] + 23023.0*z2[i]*r12[i] - 429.0*r14[i]) + 126233085.0*z16[i] - 463991880.0*z14[i]*r2[i] + 695987820.0*z12[i]*r4[i] - 548354040.0*z10[i]*r6[i] + 243221550.0*z8[i]*r8[i] - 60386040.0*z6[i]*r10[i] + 7827820.0*z4[i]*r12[i] - 447304.0*z2[i]*r14[i] + 7293.0*r16[i]);
      prCofDX[totalAN*375+i] = -0.0566912029031131*x[i]*y[i]*(23881935.0*z16[i] - 81880920.0*z14[i]*r2[i] + 112896420.0*z12[i]*r4[i] - 80120040.0*z10[i]*r6[i] + 31081050.0*z8[i]*r8[i] - 6446440.0*z6[i]*r10[i] + 644644.0*z4[i]*r12[i] - 24024.0*z2[i]*r14[i] + 143.0*r16[i]);
      prCofDX[totalAN*376+i] = 0.00919334476036647*x[i]*z[i]*(-119409675.0*z16[i] + 463991880.0*z14[i]*r2[i] - 738168900.0*z12[i]*r4[i] + 619109400.0*z10[i]*r6[i] - 293543250.0*z8[i]*r8[i] + 78278200.0*z6[i]*r10[i] - 10958948.0*z4[i]*r12[i] + 680680.0*z2[i]*r14[i] - 12155.0*r16[i]);
      prCofDX[totalAN*377+i] = -0.0566912029031131*x2[i]*(23881935.0*z16[i] - 81880920.0*z14[i]*r2[i] + 112896420.0*z12[i]*r4[i] - 80120040.0*z10[i]*r6[i] + 31081050.0*z8[i]*r8[i] - 6446440.0*z6[i]*r10[i] + 644644.0*z4[i]*r12[i] - 24024.0*z2[i]*r14[i] + 143.0*r16[i]) + 163706.333476296*z18[i] - 676947.811401979*z16[i]*r2[i] + 1160481.96240339*z14[i]*r4[i] - 1066705.64220918*z12[i]*r6[i] + 567762.680530692*z10[i]*r8[i] - 176202.21119918*z8[i]*r10[i] + 30454.7031702287*z6[i]*r12[i] - 2610.40312887675*z4[i]*r14[i] + 85.1218411590243*z2[i]*r16[i] - 0.45038011195251*r18[i];
      prCofDX[totalAN*378+i] = 0.00240131363330655*x[i]*z[i]*(126233085.0*z16[i] - 463991880.0*z14[i]*r2[i] + 695987820.0*z12[i]*r4[i] - 548354040.0*z10[i]*r6[i] + 243221550.0*z8[i]*r8[i] - 60386040.0*z6[i]*r10[i] + 7827820.0*z4[i]*r12[i] - 447304.0*z2[i]*r14[i] - 136.0*(x2[i] - y2[i])*(3411705.0*z14[i] - 10235115.0*z12[i]*r2[i] + 12096045.0*z10[i]*r4[i] - 7153575.0*z8[i]*r6[i] + 2220075.0*z6[i]*r8[i] - 345345.0*z4[i]*r10[i] + 23023.0*z2[i]*r12[i] - 429.0*r14[i]) + 7293.0*r16[i]);
      prCofDX[totalAN*379+i] = -0.557270346156495*x2[i]*(x2[i] - 3.0*y2[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]) + 0.0348293966347809*(x2[i] - y2[i])*(11475735.0*z16[i] - 37218600.0*z14[i]*r2[i] + 48384180.0*z12[i]*r4[i] - 32256120.0*z10[i]*r6[i] + 11705850.0*z8[i]*r8[i] - 2260440.0*z6[i]*r10[i] + 209300.0*z4[i]*r12[i] - 7176.0*z2[i]*r14[i] + 39.0*r16[i]);
      prCofDX[totalAN*380+i] = 0.445429115517452*x[i]*z[i]*(2.0*(x2[i] - 3.0*y2[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]) - 3.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]));
      prCofDX[totalAN*381+i] = -0.915570851773108*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]) + 0.176071317648675*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i]);
      prCofDX[totalAN*382+i] = 0.564684468239301*x[i]*z[i]*((x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i]) - 26.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]));
      prCofDX[totalAN*383+i] = -0.798584433445459*x2[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]) + 0.465840919509851*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i]);
      prCofDX[totalAN*384+i] = 0.266194811148486*x[i]*z[i]*(4.0*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i]) - (x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]));
      prCofDX[totalAN*385+i] = -0.955574995064948*x2[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i]) + 0.477787497532474*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i]);
      prCofDX[totalAN*386+i] = 0.361619017612871*x[i]*z[i]*(-36.0*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) + 5.0*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i]));
      prCofDX[totalAN*387+i] = -3.96133786338855*x2[i]*(2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) + 5.44683956215926*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDX[totalAN*388+i] = 46.7874082919139*x[i]*z[i]*(-(77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]) + 2.0*(407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDX[totalAN*389+i] = -3.12611521149014*x2[i]*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]) + 6.7732496248953*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDX[totalAN*390+i] = 4.8875933516657*x[i]*z[i]*(-2.0*(-3.0*x2[i] - 3.0*y2[i] + 32.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]) + 7.0*(259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]));
      prCofDX[totalAN*391+i] = -2.24916778827046*x2[i]*(-x2[i] - y2[i] + 34.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]) + 2.81145973533808*(1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      prCofDX[totalAN*392+i] = 4.43541869366192*x[i]*z[i]*(-3.0*x16[i] + 360.0*x14[i]*y2[i] - 5460.0*x12[i]*y4[i] + 24024.0*x10[i]*y6[i] - 38610.0*x8[i]*y8[i] + 24024.0*x6[i]*y10[i] - 5460.0*x4[i]*y12[i] + 360.0*x2[i]*y14[i] - 3.0*y16[i] + 8.0*(-3.0*x2[i] - 3.0*y2[i] + 34.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]));
      prCofDX[totalAN*393+i] = -1.2803950883772*x2[i]*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]) - 10.8833582512062*(x2[i] + y2[i] - 36.0*z2[i])*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]);
      prCofDX[totalAN*394+i] = 99.1293751849004*x[i]*z[i]*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]);
      prCofDX[totalAN*395+i] = 16.974291902649*x18[i] - 2597.0666611053*x16[i]*y2[i] + 51941.3332221061*x14[i]*y4[i] - 315110.754880777*x12[i]*y6[i] + 742761.065076117*x10[i]*y8[i] - 742761.065076117*x8[i]*y10[i] + 315110.754880777*x6[i]*y12[i] - 51941.3332221061*x4[i]*y14[i] + 2597.0666611053*x2[i]*y16[i] - 16.974291902649*y18[i];

      prCofDY[totalAN*357+i] = 16.974291902649*x18[i] - 2597.0666611053*x16[i]*y2[i] + 51941.3332221061*x14[i]*y4[i] - 315110.754880777*x12[i]*y6[i] + 742761.065076117*x10[i]*y8[i] - 742761.065076117*x8[i]*y10[i] + 315110.754880777*x6[i]*y12[i] - 51941.3332221061*x4[i]*y14[i] + 2597.0666611053*x2[i]*y16[i] - 16.974291902649*y18[i];
      prCofDY[totalAN*358+i] = 99.1293751849004*x[i]*z[i]*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]);
      prCofDY[totalAN*359+i] = -1.2803950883772*y2[i]*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]) - 10.8833582512062*(x2[i] + y2[i] - 36.0*z2[i])*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]);
      prCofDY[totalAN*360+i] = 35.4833495492953*x[i]*z[i]*(-6.0*y2[i]*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]) + (-3.0*x2[i] - 3.0*y2[i] + 34.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]));
      prCofDY[totalAN*361+i] = -2.24916778827046*y2[i]*(-x2[i] - y2[i] + 34.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]) + 2.81145973533808*(1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      prCofDY[totalAN*362+i] = 4.8875933516657*x[i]*z[i]*(-4.0*y2[i]*(-3.0*x2[i] - 3.0*y2[i] + 32.0*z2[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]) + 7.0*(259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]));
      prCofDY[totalAN*363+i] = -3.12611521149014*y2[i]*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]) + 6.7732496248953*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDY[totalAN*364+i] = 93.5748165838277*x[i]*z[i]*(-2.0*y2[i]*(77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]) + (407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]));
      prCofDY[totalAN*365+i] = -3.96133786338855*y2[i]*(2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 5.44683956215926*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDY[totalAN*366+i] = 0.361619017612871*x[i]*z[i]*(-72.0*y2[i]*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i]) + 5.0*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i]));
      prCofDY[totalAN*367+i] = -0.955574995064948*y2[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i]) + 0.477787497532474*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i]);
      prCofDY[totalAN*368+i] = 1.06477924459395*x[i]*z[i]*(-2.0*y2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]) + (x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i]));
      prCofDY[totalAN*369+i] = -0.798584433445459*y2[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]) + 0.465840919509851*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i]);
      prCofDY[totalAN*370+i] = 0.564684468239301*x[i]*z[i]*(-52.0*y2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]) + (x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i]));
      prCofDY[totalAN*371+i] = -0.915570851773108*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]) + 0.176071317648675*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i]);
      prCofDY[totalAN*372+i] = 0.890858231034903*x[i]*z[i]*(-6.0*y2[i]*(x2[i] - y2[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]) + (x2[i] - 3.0*y2[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]));
      prCofDY[totalAN*373+i] = -0.557270346156495*y2[i]*(3.0*x2[i] - y2[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]) + 0.0348293966347809*(x2[i] - y2[i])*(11475735.0*z16[i] - 37218600.0*z14[i]*r2[i] + 48384180.0*z12[i]*r4[i] - 32256120.0*z10[i]*r6[i] + 11705850.0*z8[i]*r8[i] - 2260440.0*z6[i]*r10[i] + 209300.0*z4[i]*r12[i] - 7176.0*z2[i]*r14[i] + 39.0*r16[i]);
      prCofDY[totalAN*374+i] = 0.00240131363330655*x[i]*z[i]*(-272.0*y2[i]*(3411705.0*z14[i] - 10235115.0*z12[i]*r2[i] + 12096045.0*z10[i]*r4[i] - 7153575.0*z8[i]*r6[i] + 2220075.0*z6[i]*r8[i] - 345345.0*z4[i]*r10[i] + 23023.0*z2[i]*r12[i] - 429.0*r14[i]) + 126233085.0*z16[i] - 463991880.0*z14[i]*r2[i] + 695987820.0*z12[i]*r4[i] - 548354040.0*z10[i]*r6[i] + 243221550.0*z8[i]*r8[i] - 60386040.0*z6[i]*r10[i] + 7827820.0*z4[i]*r12[i] - 447304.0*z2[i]*r14[i] + 7293.0*r16[i]);
      prCofDY[totalAN*375+i] = -0.0566912029031131*y2[i]*(23881935.0*z16[i] - 81880920.0*z14[i]*r2[i] + 112896420.0*z12[i]*r4[i] - 80120040.0*z10[i]*r6[i] + 31081050.0*z8[i]*r8[i] - 6446440.0*z6[i]*r10[i] + 644644.0*z4[i]*r12[i] - 24024.0*z2[i]*r14[i] + 143.0*r16[i]) + 163706.333476296*z18[i] - 676947.811401979*z16[i]*r2[i] + 1160481.96240339*z14[i]*r4[i] - 1066705.64220918*z12[i]*r6[i] + 567762.680530692*z10[i]*r8[i] - 176202.21119918*z8[i]*r10[i] + 30454.7031702287*z6[i]*r12[i] - 2610.40312887675*z4[i]*r14[i] + 85.1218411590243*z2[i]*r16[i] - 0.45038011195251*r18[i];
      prCofDY[totalAN*376+i] = 0.00919334476036647*y[i]*z[i]*(-119409675.0*z16[i] + 463991880.0*z14[i]*r2[i] - 738168900.0*z12[i]*r4[i] + 619109400.0*z10[i]*r6[i] - 293543250.0*z8[i]*r8[i] + 78278200.0*z6[i]*r10[i] - 10958948.0*z4[i]*r12[i] + 680680.0*z2[i]*r14[i] - 12155.0*r16[i]);
      prCofDY[totalAN*377+i] = -0.0566912029031131*x[i]*y[i]*(23881935.0*z16[i] - 81880920.0*z14[i]*r2[i] + 112896420.0*z12[i]*r4[i] - 80120040.0*z10[i]*r6[i] + 31081050.0*z8[i]*r8[i] - 6446440.0*z6[i]*r10[i] + 644644.0*z4[i]*r12[i] - 24024.0*z2[i]*r14[i] + 143.0*r16[i]);
      prCofDY[totalAN*378+i] = 0.00240131363330655*y[i]*z[i]*(-126233085.0*z16[i] + 463991880.0*z14[i]*r2[i] - 695987820.0*z12[i]*r4[i] + 548354040.0*z10[i]*r6[i] - 243221550.0*z8[i]*r8[i] + 60386040.0*z6[i]*r10[i] - 7827820.0*z4[i]*r12[i] + 447304.0*z2[i]*r14[i] - 136.0*(x2[i] - y2[i])*(3411705.0*z14[i] - 10235115.0*z12[i]*r2[i] + 12096045.0*z10[i]*r4[i] - 7153575.0*z8[i]*r6[i] + 2220075.0*z6[i]*r8[i] - 345345.0*z4[i]*r10[i] + 23023.0*z2[i]*r12[i] - 429.0*r14[i]) - 7293.0*r16[i]);
      prCofDY[totalAN*379+i] = 0.0696587932695619*x[i]*y[i]*(-11475735.0*z16[i] + 37218600.0*z14[i]*r2[i] - 48384180.0*z12[i]*r4[i] + 32256120.0*z10[i]*r6[i] - 11705850.0*z8[i]*r8[i] + 2260440.0*z6[i]*r10[i] - 209300.0*z4[i]*r12[i] + 7176.0*z2[i]*r14[i] - 8.0*(x2[i] - 3.0*y2[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]) - 39.0*r16[i]);
      prCofDY[totalAN*380+i] = -0.445429115517452*y[i]*z[i]*(2.0*(3.0*x2[i] - y2[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]) + 3.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]));
      prCofDY[totalAN*381+i] = -0.0704285270594699*x[i]*y[i]*(10.0*(x2[i] - y2[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i]) + 13.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]));
      prCofDY[totalAN*382+i] = -0.564684468239301*y[i]*z[i]*((5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i]) + 26.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]));
      prCofDY[totalAN*383+i] = -0.133097405574243*x[i]*y[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i]) + 6.0*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]));
      prCofDY[totalAN*384+i] = -0.266194811148486*y[i]*z[i]*(4.0*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i]) + (x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]));
      prCofDY[totalAN*385+i] = -0.955574995064948*x[i]*y[i]*(4.0*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i]) + (x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i]));
      prCofDY[totalAN*386+i] = -0.361619017612871*y[i]*z[i]*(36.0*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) + 5.0*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i]));
      prCofDY[totalAN*387+i] = -0.990334465847138*x[i]*y[i]*(4.0*(2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) + 11.0*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i])*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i]));
      prCofDY[totalAN*388+i] = -46.7874082919139*y[i]*z[i]*((77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]) + 2.0*(407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]));
      prCofDY[totalAN*389+i] = -1.04203840383005*x[i]*y[i]*(3.0*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]) + 26.0*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]));
      prCofDY[totalAN*390+i] = 4.8875933516657*y[i]*z[i]*(2.0*(3.0*x2[i] + 3.0*y2[i] - 32.0*z2[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]) - 7.0*(259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]));
      prCofDY[totalAN*391+i] = 1.12458389413523*x[i]*y[i]*(2.0*(x2[i] + y2[i] - 34.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]) - 5.0*(1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]));
      prCofDY[totalAN*392+i] = 4.43541869366192*y[i]*z[i]*(-3.0*x16[i] + 360.0*x14[i]*y2[i] - 5460.0*x12[i]*y4[i] + 24024.0*x10[i]*y6[i] - 38610.0*x8[i]*y8[i] + 24024.0*x6[i]*y10[i] - 5460.0*x4[i]*y12[i] + 360.0*x2[i]*y14[i] - 3.0*y16[i] - 8.0*(-3.0*x2[i] - 3.0*y2[i] + 34.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]));
      prCofDY[totalAN*393+i] = 1.2803950883772*x[i]*y[i]*(-x16[i] + 136.0*x14[i]*y2[i] - 2380.0*x12[i]*y4[i] + 12376.0*x10[i]*y6[i] - 24310.0*x8[i]*y8[i] + 19448.0*x6[i]*y10[i] - 6188.0*x4[i]*y12[i] + 680.0*x2[i]*y14[i] - 17.0*y16[i] + 136.0*(x2[i] + y2[i] - 36.0*z2[i])*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]));
      prCofDY[totalAN*394+i] = -99.1293751849004*y[i]*z[i]*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]);
      prCofDY[totalAN*395+i] = 33.9485838052981*x[i]*y[i]*(-9.0*x16[i] + 408.0*x14[i]*y2[i] - 4284.0*x12[i]*y4[i] + 15912.0*x10[i]*y6[i] - 24310.0*x8[i]*y8[i] + 15912.0*x6[i]*y10[i] - 4284.0*x4[i]*y12[i] + 408.0*x2[i]*y14[i] - 9.0*y16[i]);

      prCofDZ[totalAN*357+i] = 0;
      prCofDZ[totalAN*358+i] = 11.0143750205445*x[i]*y[i]*(9.0*x16[i] - 408.0*x14[i]*y2[i] + 4284.0*x12[i]*y4[i] - 15912.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 15912.0*x6[i]*y10[i] + 4284.0*x4[i]*y12[i] - 408.0*x2[i]*y14[i] + 9.0*y16[i]);
      prCofDZ[totalAN*359+i] = 46.0942231815793*y[i]*z[i]*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]);
      prCofDZ[totalAN*360+i] = -106.450048647886*x[i]*y[i]*(x2[i] + y2[i] - 34.0*z2[i])*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]);
      prCofDZ[totalAN*361+i] = 25.4905682670652*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 32.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
      prCofDZ[totalAN*362+i] = 14.6627800549971*x[i]*y[i]*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
      prCofDZ[totalAN*363+i] = 100.035686767684*y[i]*z[i]*(77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*364+i] = 31.1916055279426*x[i]*y[i]*(2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
      prCofDZ[totalAN*365+i] = 23.7680271803313*y[i]*z[i]*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*366+i] = 3.25457115851584*x[i]*y[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i])*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i]);
      prCofDZ[totalAN*367+i] = 0.424699997806643*y[i]*z[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]);
      prCofDZ[totalAN*368+i] = 9.5830132013455*x[i]*y[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]);
      prCofDZ[totalAN*369+i] = 20.7631952695819*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]);
      prCofDZ[totalAN*370+i] = 2.44696602903697*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]);
      prCofDZ[totalAN*371+i] = 1.69028464942728*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]);
      prCofDZ[totalAN*372+i] = 2.67257469310471*x[i]*y[i]*(x2[i] - y2[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]);
      prCofDZ[totalAN*373+i] = 0.37151356410433*y[i]*z[i]*(3.0*x2[i] - y2[i])*(3411705.0*z14[i] - 10235115.0*z12[i]*r2[i] + 12096045.0*z10[i]*r4[i] - 7153575.0*z8[i]*r6[i] + 2220075.0*z6[i]*r8[i] - 345345.0*z4[i]*r10[i] + 23023.0*z2[i]*r12[i] - 429.0*r14[i]);
      prCofDZ[totalAN*374+i] = 0.122466995298634*x[i]*y[i]*(23881935.0*z16[i] - 81880920.0*z14[i]*r2[i] + 112896420.0*z12[i]*r4[i] - 80120040.0*z10[i]*r6[i] + 31081050.0*z8[i]*r8[i] - 6446440.0*z6[i]*r10[i] + 644644.0*z4[i]*r12[i] - 24024.0*z2[i]*r14[i] + 143.0*r16[i]);
      prCofDZ[totalAN*375+i] = 0.0133391065654384*y[i]*z[i]*(119409675.0*z16[i] - 463991880.0*z14[i]*r2[i] + 738168900.0*z12[i]*r4[i] - 619109400.0*z10[i]*r6[i] + 293543250.0*z8[i]*r8[i] - 78278200.0*z6[i]*r10[i] + 10958948.0*z4[i]*r12[i] - 680680.0*z2[i]*r14[i] + 12155.0*r16[i]);
      prCofDZ[totalAN*376+i] = 1158761.77166489*z18[i] - 5065444.31613507*z16[i]*r2[i] + 9209898.75660922*z14[i]*r4[i] - 9011836.41775741*z12[i]*r6[i] + 5127424.16872404*z10[i]*r8[i] - 1709141.38957468*z8[i]*r10[i] + 319039.72605394*z6[i]*r12[i] - 29724.1980795597*z4[i]*r14[i] + 1061.57850284142*z2[i]*r16[i] - 6.20806142012525*r18[i];
      prCofDZ[totalAN*377+i] = 0.0133391065654384*x[i]*z[i]*(119409675.0*z16[i] - 463991880.0*z14[i]*r2[i] + 738168900.0*z12[i]*r4[i] - 619109400.0*z10[i]*r6[i] + 293543250.0*z8[i]*r8[i] - 78278200.0*z6[i]*r10[i] + 10958948.0*z4[i]*r12[i] - 680680.0*z2[i]*r14[i] + 12155.0*r16[i]);
      prCofDZ[totalAN*378+i] = 0.0612334976493172*(x2[i] - y2[i])*(23881935.0*z16[i] - 81880920.0*z14[i]*r2[i] + 112896420.0*z12[i]*r4[i] - 80120040.0*z10[i]*r6[i] + 31081050.0*z8[i]*r8[i] - 6446440.0*z6[i]*r10[i] + 644644.0*z4[i]*r12[i] - 24024.0*z2[i]*r14[i] + 143.0*r16[i]);
      prCofDZ[totalAN*379+i] = 0.37151356410433*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(3411705.0*z14[i] - 10235115.0*z12[i]*r2[i] + 12096045.0*z10[i]*r4[i] - 7153575.0*z8[i]*r6[i] + 2220075.0*z6[i]*r8[i] - 345345.0*z4[i]*r10[i] + 23023.0*z2[i]*r12[i] - 429.0*r14[i]);
      prCofDZ[totalAN*380+i] = 0.668143673276177*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(1550775.0*z14[i] - 4032015.0*z12[i]*r2[i] + 4032015.0*z10[i]*r4[i] - 1950975.0*z8[i]*r6[i] + 470925.0*z6[i]*r8[i] - 52325.0*z4[i]*r10[i] + 2093.0*z2[i]*r12[i] - 13.0*r14[i]);
      prCofDZ[totalAN*381+i] = 1.69028464942728*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(471975.0*z12[i] - 1051830.0*z10[i]*r2[i] + 876525.0*z8[i]*r4[i] - 339300.0*z6[i]*r6[i] + 61425.0*z4[i]*r8[i] - 4550.0*z2[i]*r10[i] + 91.0*r12[i]);
      prCofDZ[totalAN*382+i] = 1.22348301451849*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(471975.0*z12[i] - 890010.0*z10[i]*r2[i] + 606825.0*z8[i]*r4[i] - 182700.0*z6[i]*r6[i] + 23625.0*z4[i]*r8[i] - 1050.0*z2[i]*r10[i] + 7.0*r12[i]);
      prCofDZ[totalAN*383+i] = 20.7631952695819*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(18879.0*z10[i] - 29667.0*z8[i]*r2[i] + 16182.0*z6[i]*r4[i] - 3654.0*z4[i]*r6[i] + 315.0*z2[i]*r8[i] - 7.0*r10[i]);
      prCofDZ[totalAN*384+i] = 1.19787665016819*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(207669.0*z10[i] - 267003.0*z8[i]*r2[i] + 113274.0*z6[i]*r4[i] - 18270.0*z4[i]*r6[i] + 945.0*z2[i]*r8[i] - 7.0*r10[i]);
      prCofDZ[totalAN*385+i] = 0.424699997806643*x[i]*z[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(346115.0*z8[i] - 356004.0*z6[i]*r2[i] + 113274.0*z4[i]*r4[i] - 12180.0*z2[i]*r6[i] + 315.0*r8[i]);
      prCofDZ[totalAN*386+i] = 1.62728557925792*(49445.0*z8[i] - 39556.0*z6[i]*r2[i] + 8990.0*z4[i]*r4[i] - 580.0*z2[i]*r6[i] + 5.0*r8[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]);
      prCofDZ[totalAN*387+i] = 23.7680271803313*x[i]*z[i]*(1705.0*z6[i] - 1023.0*z4[i]*r2[i] + 155.0*z2[i]*r4[i] - 5.0*r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
      prCofDZ[totalAN*388+i] = 7.79790138198564*(2387.0*z6[i] - 1023.0*z4[i]*r2[i] + 93.0*z2[i]*r4[i] - r6[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
      prCofDZ[totalAN*389+i] = 100.035686767684*x[i]*z[i]*(77.0*z4[i] - 22.0*z2[i]*r2[i] + r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
      prCofDZ[totalAN*390+i] = 7.33139002749855*(385.0*z4[i] - 66.0*z2[i]*r2[i] + r4[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
      prCofDZ[totalAN*391+i] = 25.4905682670652*x[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 32.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
      prCofDZ[totalAN*392+i] = -6.65312804049287*(x2[i] + y2[i] - 34.0*z2[i])*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]);
      prCofDZ[totalAN*393+i] = 46.0942231815793*x[i]*z[i]*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]);
      prCofDZ[totalAN*394+i] = 5.50718751027224*x18[i] - 842.599689071654*x16[i]*y2[i] + 16851.9937814331*x14[i]*y4[i] - 102235.428940694*x12[i]*y6[i] + 240983.511074493*x10[i]*y8[i] - 240983.511074493*x8[i]*y10[i] + 102235.428940694*x6[i]*y12[i] - 16851.9937814331*x4[i]*y14[i] + 842.599689071654*x2[i]*y16[i] - 5.50718751027224*y18[i];
      prCofDZ[totalAN*395+i] = 0;
        }



    if (lMax > 19){ 
preCoef[totalAN*396 + i] = 3.61792858037396*x[i]*y[i]*(5.0*x18[i] - 285.0*x16[i]*y2[i] + 3876.0*x14[i]*y4[i] - 19380.0*x12[i]*y6[i] + 41990.0*x10[i]*y8[i] - 41990.0*x8[i]*y10[i] + 19380.0*x6[i]*y12[i] - 3876.0*x4[i]*y14[i] + 285.0*x2[i]*y16[i] - 5.0*y18[i]);
preCoef[totalAN*397 + i] = 5.72044736290064*y[i]*z[i]*(19.0*x18[i] - 969.0*x16[i]*y2[i] + 11628.0*x14[i]*y4[i] - 50388.0*x12[i]*y6[i] + 92378.0*x10[i]*y8[i] - 75582.0*x8[i]*y10[i] + 27132.0*x6[i]*y12[i] - 3876.0*x4[i]*y14[i] + 171.0*x2[i]*y16[i] - y18[i]);
preCoef[totalAN*398 + i] = -1.29542623480908*x[i]*y[i]*(x2[i] + y2[i] - 38.0*z2[i])*(9.0*x16[i] - 408.0*x14[i]*y2[i] + 4284.0*x12[i]*y4[i] - 15912.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 15912.0*x6[i]*y10[i] + 4284.0*x4[i]*y12[i] - 408.0*x2[i]*y14[i] + 9.0*y16[i]);
preCoef[totalAN*399 + i] = 6.91568363939543*y[i]*z[i]*(-x2[i] - y2[i] + 12.0*z2[i])*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]);
preCoef[totalAN*400 + i] = 9.09545109472669*x[i]*y[i]*(481.0*z4[i] - 74.0*z2[i]*r2[i] + r4[i])*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]);
preCoef[totalAN*401 + i] = 0.508451173345844*y[i]*z[i]*(1443.0*z4[i] - 370.0*z2[i]*r2[i] + 15.0*r4[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]);
preCoef[totalAN*402 + i] = 1.05259392999953*x[i]*y[i]*(3367.0*z6[i] - 1295.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - r6[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]);
preCoef[totalAN*403 + i] = 8.1193141272878*y[i]*z[i]*(481.0*z6[i] - 259.0*z4[i]*r2[i] + 35.0*z2[i]*r4[i] - r6[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]);
preCoef[totalAN*404 + i] = 1.99883696331483*x[i]*y[i]*(15873.0*z8[i] - 11396.0*z6[i]*r2[i] + 2310.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + r8[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]);
preCoef[totalAN*405 + i] = 2.82678234249248*y[i]*z[i]*(5291.0*z8[i] - 4884.0*z6[i]*r2[i] + 1386.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + 3.0*r8[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]);
preCoef[totalAN*406 + i] = 0.321100896851849*x[i]*y[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i])*(164021.0*z10[i] - 189255.0*z8[i]*r2[i] + 71610.0*z6[i]*r4[i] - 10230.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 3.0*r10[i]);
preCoef[totalAN*407 + i] = 0.972181244054524*y[i]*z[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(44733.0*z10[i] - 63085.0*z8[i]*r2[i] + 30690.0*z6[i]*r4[i] - 6138.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 9.0*r10[i]);
preCoef[totalAN*408 + i] = 1.25074523746207*x[i]*y[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(432419.0*z12[i] - 731786.0*z10[i]*r2[i] + 445005.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 13485.0*z4[i]*r8[i] - 522.0*z2[i]*r10[i] + 3.0*r12[i]);
preCoef[totalAN*409 + i] = 0.426119611785933*y[i]*z[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(232841.0*z12[i] - 465682.0*z10[i]*r2[i] + 346115.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 18879.0*z4[i]*r8[i] - 1218.0*z2[i]*r10[i] + 21.0*r12[i]);
preCoef[totalAN*410 + i] = 0.920523570163618*x[i]*y[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(299367.0*z14[i] - 698523.0*z12[i]*r2[i] + 623007.0*z10[i]*r4[i] - 267003.0*z8[i]*r6[i] + 56637.0*z6[i]*r8[i] - 5481.0*z4[i]*r10[i] + 189.0*z2[i]*r12[i] - r14[i]);
preCoef[totalAN*411 + i] = 0.139837568674965*y[i]*z[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(1297257.0*z14[i] - 3492615.0*z12[i]*r2[i] + 3681405.0*z10[i]*r4[i] - 1928355.0*z8[i]*r6[i] + 525915.0*z6[i]*r8[i] - 71253.0*z4[i]*r10[i] + 4095.0*z2[i]*r12[i] - 65.0*r14[i]);
preCoef[totalAN*412 + i] = 0.139837568674965*x[i]*y[i]*(x2[i] - y2[i])*(6486285.0*z16[i] - 19957800.0*z14[i]*r2[i] + 24542700.0*z12[i]*r4[i] - 15426840.0*z10[i]*r6[i] + 5259150.0*z8[i]*r8[i] - 950040.0*z6[i]*r10[i] + 81900.0*z4[i]*r12[i] - 2600.0*z2[i]*r14[i] + 13.0*r16[i]);
preCoef[totalAN*413 + i] = 0.0138459825039349*y[i]*z[i]*(3.0*x2[i] - y2[i])*(19458855.0*z16[i] - 67856520.0*z14[i]*r2[i] + 96282900.0*z12[i]*r4[i] - 71524440.0*z10[i]*r6[i] + 29801850.0*z8[i]*r8[i] - 6921720.0*z6[i]*r10[i] + 835380.0*z4[i]*r12[i] - 44200.0*z2[i]*r14[i] + 663.0*r16[i]);
preCoef[totalAN*414 + i] = 0.00408295749053327*x[i]*y[i]*(149184555.0*z18[i] - 585262485.0*z16[i]*r2[i] + 949074300.0*z14[i]*r4[i] - 822531060.0*z12[i]*r6[i] + 411265530.0*z10[i]*r8[i] - 119399670.0*z8[i]*r10[i] + 19213740.0*z6[i]*r12[i] - 1524900.0*z4[i]*r14[i] + 45747.0*z2[i]*r16[i] - 221.0*r18[i]);
preCoef[totalAN*415 + i] = 0.000199703978712595*y[i]*z[i]*(1641030105.0*z18[i] - 7195285845.0*z16[i]*r2[i] + 13223768580.0*z14[i]*r4[i] - 13223768580.0*z12[i]*r6[i] + 7814045070.0*z10[i]*r8[i] - 2772725670.0*z8[i]*r10[i] + 573667380.0*z6[i]*r12[i] - 63740820.0*z4[i]*r14[i] + 3187041.0*z2[i]*r16[i] - 46189.0*r18[i]);
preCoef[totalAN*416 + i] = 237455.874096927*z20[i] - 1156836.30970298*z18[i]*r2[i] + 2391837.23492643*z16[i]*r4[i] - 2733528.26848735*z14[i]*r6[i] + 1884477.82145719*z12[i]*r8[i] - 802422.814297898*z10[i]*r10[i] + 207523.141628767*z8[i]*r12[i] - 30744.1691301877*z6[i]*r14[i] + 2305.81268476408*z4[i]*r16[i] - 66.8351502830167*z2[i]*r18[i] + 0.318262620395318*r20[i];
preCoef[totalAN*417 + i] = 0.000199703978712595*x[i]*z[i]*(1641030105.0*z18[i] - 7195285845.0*z16[i]*r2[i] + 13223768580.0*z14[i]*r4[i] - 13223768580.0*z12[i]*r6[i] + 7814045070.0*z10[i]*r8[i] - 2772725670.0*z8[i]*r10[i] + 573667380.0*z6[i]*r12[i] - 63740820.0*z4[i]*r14[i] + 3187041.0*z2[i]*r16[i] - 46189.0*r18[i]);
preCoef[totalAN*418 + i] = 0.00204147874526664*(x2[i] - y2[i])*(149184555.0*z18[i] - 585262485.0*z16[i]*r2[i] + 949074300.0*z14[i]*r4[i] - 822531060.0*z12[i]*r6[i] + 411265530.0*z10[i]*r8[i] - 119399670.0*z8[i]*r10[i] + 19213740.0*z6[i]*r12[i] - 1524900.0*z4[i]*r14[i] + 45747.0*z2[i]*r16[i] - 221.0*r18[i]);
preCoef[totalAN*419 + i] = 0.0138459825039349*x[i]*z[i]*(x2[i] - 3.0*y2[i])*(19458855.0*z16[i] - 67856520.0*z14[i]*r2[i] + 96282900.0*z12[i]*r4[i] - 71524440.0*z10[i]*r6[i] + 29801850.0*z8[i]*r8[i] - 6921720.0*z6[i]*r10[i] + 835380.0*z4[i]*r12[i] - 44200.0*z2[i]*r14[i] + 663.0*r16[i]);
preCoef[totalAN*420 + i] = 0.0349593921687412*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(6486285.0*z16[i] - 19957800.0*z14[i]*r2[i] + 24542700.0*z12[i]*r4[i] - 15426840.0*z10[i]*r6[i] + 5259150.0*z8[i]*r8[i] - 950040.0*z6[i]*r10[i] + 81900.0*z4[i]*r12[i] - 2600.0*z2[i]*r14[i] + 13.0*r16[i]);
preCoef[totalAN*421 + i] = 0.139837568674965*x[i]*z[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(1297257.0*z14[i] - 3492615.0*z12[i]*r2[i] + 3681405.0*z10[i]*r4[i] - 1928355.0*z8[i]*r6[i] + 525915.0*z6[i]*r8[i] - 71253.0*z4[i]*r10[i] + 4095.0*z2[i]*r12[i] - 65.0*r14[i]);
preCoef[totalAN*422 + i] = 0.460261785081809*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(299367.0*z14[i] - 698523.0*z12[i]*r2[i] + 623007.0*z10[i]*r4[i] - 267003.0*z8[i]*r6[i] + 56637.0*z6[i]*r8[i] - 5481.0*z4[i]*r10[i] + 189.0*z2[i]*r12[i] - r14[i]);
preCoef[totalAN*423 + i] = 0.426119611785933*x[i]*z[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(232841.0*z12[i] - 465682.0*z10[i]*r2[i] + 346115.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 18879.0*z4[i]*r8[i] - 1218.0*z2[i]*r10[i] + 21.0*r12[i]);
preCoef[totalAN*424 + i] = 0.156343154682758*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(432419.0*z12[i] - 731786.0*z10[i]*r2[i] + 445005.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 13485.0*z4[i]*r8[i] - 522.0*z2[i]*r10[i] + 3.0*r12[i]);
preCoef[totalAN*425 + i] = 0.972181244054524*x[i]*z[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(44733.0*z10[i] - 63085.0*z8[i]*r2[i] + 30690.0*z6[i]*r4[i] - 6138.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 9.0*r10[i]);
preCoef[totalAN*426 + i] = 0.160550448425925*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i])*(164021.0*z10[i] - 189255.0*z8[i]*r2[i] + 71610.0*z6[i]*r4[i] - 10230.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 3.0*r10[i]);
preCoef[totalAN*427 + i] = 2.82678234249248*x[i]*z[i]*(5291.0*z8[i] - 4884.0*z6[i]*r2[i] + 1386.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + 3.0*r8[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]);
preCoef[totalAN*428 + i] = 0.499709240828707*(15873.0*z8[i] - 11396.0*z6[i]*r2[i] + 2310.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + r8[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]);
preCoef[totalAN*429 + i] = 8.1193141272878*x[i]*z[i]*(481.0*z6[i] - 259.0*z4[i]*r2[i] + 35.0*z2[i]*r4[i] - r6[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]);
preCoef[totalAN*430 + i] = 0.526296964999764*(3367.0*z6[i] - 1295.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - r6[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]);
preCoef[totalAN*431 + i] = 0.508451173345844*x[i]*z[i]*(1443.0*z4[i] - 370.0*z2[i]*r2[i] + 15.0*r4[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]);
preCoef[totalAN*432 + i] = 0.568465693420418*(481.0*z4[i] - 74.0*z2[i]*r2[i] + r4[i])*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]);
preCoef[totalAN*433 + i] = 6.91568363939543*x[i]*z[i]*(-x2[i] - y2[i] + 12.0*z2[i])*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]);
preCoef[totalAN*434 + i] = -0.647713117404541*(x2[i] + y2[i] - 38.0*z2[i])*(x18[i] - 153.0*x16[i]*y2[i] + 3060.0*x14[i]*y4[i] - 18564.0*x12[i]*y6[i] + 43758.0*x10[i]*y8[i] - 43758.0*x8[i]*y10[i] + 18564.0*x6[i]*y12[i] - 3060.0*x4[i]*y14[i] + 153.0*x2[i]*y16[i] - y18[i]);
preCoef[totalAN*435 + i] = 5.72044736290064*x[i]*z[i]*(x18[i] - 171.0*x16[i]*y2[i] + 3876.0*x14[i]*y4[i] - 27132.0*x12[i]*y6[i] + 75582.0*x10[i]*y8[i] - 92378.0*x8[i]*y10[i] + 50388.0*x6[i]*y12[i] - 11628.0*x4[i]*y14[i] + 969.0*x2[i]*y16[i] - 19.0*y18[i]);
preCoef[totalAN*436 + i] = 0.904482145093491*x20[i] - 171.851607567763*x18[i]*y2[i] + 4382.21599297796*x16[i]*y4[i] - 35057.7279438237*x14[i]*y6[i] + 113937.615817427*x12[i]*y8[i] - 167108.503198893*x10[i]*y10[i] + 113937.615817427*x8[i]*y12[i] - 35057.7279438237*x6[i]*y14[i] + 4382.21599297796*x4[i]*y16[i] - 171.851607567763*x2[i]*y18[i] + 0.904482145093491*y20[i];
        if(return_derivatives){

prCofDX[totalAN*396+i] = 18.0896429018698*y[i]*(19.0*x18[i] - 969.0*x16[i]*y2[i] + 11628.0*x14[i]*y4[i] - 50388.0*x12[i]*y6[i] + 92378.0*x10[i]*y8[i] - 75582.0*x8[i]*y10[i] + 27132.0*x6[i]*y12[i] - 3876.0*x4[i]*y14[i] + 171.0*x2[i]*y16[i] - y18[i]) ;
prCofDX[totalAN*397+i] = 217.376999790224*x[i]*y[i]*z[i]*(9.0*x16[i] - 408.0*x14[i]*y2[i] + 4284.0*x12[i]*y4[i] - 15912.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 15912.0*x6[i]*y10[i] + 4284.0*x4[i]*y12[i] - 408.0*x2[i]*y14[i] + 9.0*y16[i]) ;
prCofDX[totalAN*398+i] = 1.29542623480908*y[i]*(x2[i]*(-18.0*x16[i] + 816.0*x14[i]*y2[i] - 8568.0*x12[i]*y4[i] + 31824.0*x10[i]*y6[i] - 48620.0*x8[i]*y8[i] + 31824.0*x6[i]*y10[i] - 8568.0*x4[i]*y12[i] + 816.0*x2[i]*y14[i] - 18.0*y16[i]) - 9.0*(x2[i] + y2[i] - 38.0*z2[i])*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i])) ;
prCofDX[totalAN*399+i] = 13.8313672787909*x[i]*y[i]*z[i]*(-17.0*x16[i] + 680.0*x14[i]*y2[i] - 6188.0*x12[i]*y4[i] + 19448.0*x10[i]*y6[i] - 24310.0*x8[i]*y8[i] + 12376.0*x6[i]*y10[i] - 2380.0*x4[i]*y12[i] + 136.0*x2[i]*y14[i] - y16[i] + 136.0*(-x2[i] - y2[i] + 12.0*z2[i])*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i])) ;
prCofDX[totalAN*400+i] = 9.09545109472669*y[i]*(-4.0*x2[i]*(-x2[i] - y2[i] + 36.0*z2[i])*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]) + (481.0*z4[i] - 74.0*z2[i]*r2[i] + r4[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i])) ;
prCofDX[totalAN*401+i] = 5.08451173345844*x[i]*y[i]*z[i]*(-2.0*(-3.0*x2[i] - 3.0*y2[i] + 34.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]) + 3.0*(1443.0*z4[i] - 370.0*z2[i]*r2[i] + 15.0*r4[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i])) ;
prCofDX[totalAN*402+i] = 1.05259392999953*y[i]*(-2.0*x2[i]*(1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]) + 7.0*(3367.0*z6[i] - 1295.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - r6[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i])) ;
prCofDX[totalAN*403+i] = 16.2386282545756*x[i]*y[i]*z[i]*(-(259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]) + 26.0*(481.0*z6[i] - 259.0*z4[i]*r2[i] + 35.0*z2[i]*r4[i] - r6[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i])) ;
prCofDX[totalAN*404+i] = 1.99883696331483*y[i]*(-8.0*x2[i]*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]) + 3.0*(15873.0*z8[i] - 11396.0*z6[i]*r2[i] + 2310.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + r8[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i])) ;
prCofDX[totalAN*405+i] = 5.65356468498497*x[i]*y[i]*z[i]*(-12.0*(407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 11.0*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i])*(5291.0*z8[i] - 4884.0*z6[i]*r2[i] + 1386.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + 3.0*r8[i])) ;
prCofDX[totalAN*406+i] = 1.60550448425925*y[i]*(-6.0*x2[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i])*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i]) + (9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(164021.0*z10[i] - 189255.0*z8[i]*r2[i] + 71610.0*z6[i]*r4[i] - 10230.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 3.0*r10[i])) ;
prCofDX[totalAN*407+i] = 1.94436248810905*x[i]*y[i]*z[i]*(36.0*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(44733.0*z10[i] - 63085.0*z8[i]*r2[i] + 30690.0*z6[i]*r4[i] - 6138.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 9.0*r10[i]) - (9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i])) ;
prCofDX[totalAN*408+i] = 1.25074523746207*y[i]*(-4.0*x2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i]) + (7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(432419.0*z12[i] - 731786.0*z10[i]*r2[i] + 445005.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 13485.0*z4[i]*r8[i] - 522.0*z2[i]*r10[i] + 3.0*r12[i])) ;
prCofDX[totalAN*409+i] = 0.852239223571866*x[i]*y[i]*z[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(232841.0*z12[i] - 465682.0*z10[i]*r2[i] + 346115.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 18879.0*z4[i]*r8[i] - 1218.0*z2[i]*r10[i] + 21.0*r12[i]) - 2.0*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i])) ;
prCofDX[totalAN*410+i] = 0.920523570163618*y[i]*(-2.0*x2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i]) + 3.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(299367.0*z14[i] - 698523.0*z12[i]*r2[i] + 623007.0*z10[i]*r4[i] - 267003.0*z8[i]*r6[i] + 56637.0*z6[i]*r8[i] - 5481.0*z4[i]*r10[i] + 189.0*z2[i]*r12[i] - r14[i])) ;
prCofDX[totalAN*411+i] = 1.39837568674965*x[i]*y[i]*z[i]*(2.0*(x2[i] - y2[i])*(1297257.0*z14[i] - 3492615.0*z12[i]*r2[i] + 3681405.0*z10[i]*r4[i] - 1928355.0*z8[i]*r6[i] + 525915.0*z6[i]*r8[i] - 71253.0*z4[i]*r10[i] + 4095.0*z2[i]*r12[i] - 65.0*r14[i]) - (5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i])) ;
prCofDX[totalAN*412+i] = 0.139837568674965*y[i]*(-16.0*x2[i]*(x2[i] - y2[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i]) + (3.0*x2[i] - y2[i])*(6486285.0*z16[i] - 19957800.0*z14[i]*r2[i] + 24542700.0*z12[i]*r4[i] - 15426840.0*z10[i]*r6[i] + 5259150.0*z8[i]*r8[i] - 950040.0*z6[i]*r10[i] + 81900.0*z4[i]*r12[i] - 2600.0*z2[i]*r14[i] + 13.0*r16[i])) ;
prCofDX[totalAN*413+i] = 0.0276919650078697*x[i]*y[i]*z[i]*(58376565.0*z16[i] - 203569560.0*z14[i]*r2[i] + 288848700.0*z12[i]*r4[i] - 214573320.0*z10[i]*r6[i] + 89405550.0*z8[i]*r8[i] - 20765160.0*z6[i]*r10[i] + 2506140.0*z4[i]*r12[i] - 132600.0*z2[i]*r14[i] - 136.0*(3.0*x2[i] - y2[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]) + 1989.0*r16[i]) ;
prCofDX[totalAN*414+i] = 0.00408295749053327*y[i]*(-102.0*x2[i]*(11475735.0*z16[i] - 37218600.0*z14[i]*r2[i] + 48384180.0*z12[i]*r4[i] - 32256120.0*z10[i]*r6[i] + 11705850.0*z8[i]*r8[i] - 2260440.0*z6[i]*r10[i] + 209300.0*z4[i]*r12[i] - 7176.0*z2[i]*r14[i] + 39.0*r16[i]) + 149184555.0*z18[i] - 585262485.0*z16[i]*r2[i] + 949074300.0*z14[i]*r4[i] - 822531060.0*z12[i]*r6[i] + 411265530.0*z10[i]*r8[i] - 119399670.0*z8[i]*r10[i] + 19213740.0*z6[i]*r12[i] - 1524900.0*z4[i]*r14[i] + 45747.0*z2[i]*r16[i] - 221.0*r18[i]) ;
prCofDX[totalAN*415+i] = -0.0227662535732358*x[i]*y[i]*z[i]*(126233085.0*z16[i] - 463991880.0*z14[i]*r2[i] + 695987820.0*z12[i]*r4[i] - 548354040.0*z10[i]*r6[i] + 243221550.0*z8[i]*r8[i] - 60386040.0*z6[i]*r10[i] + 7827820.0*z4[i]*r12[i] - 447304.0*z2[i]*r14[i] + 7293.0*r16[i]) ;
prCofDX[totalAN*416+i] = 0.00261836791769081*x[i]*(-883631595.0*z18[i] + 3653936055.0*z16[i]*r2[i] - 6263890380.0*z14[i]*r4[i] + 5757717420.0*z12[i]*r6[i] - 3064591530.0*z10[i]*r8[i] + 951080130.0*z8[i]*r10[i] - 164384220.0*z6[i]*r12[i] + 14090076.0*z4[i]*r14[i] - 459459.0*z2[i]*r16[i] + 2431.0*r18[i]) ;
prCofDX[totalAN*417+i] = 0.000199703978712595*z[i]*(-114.0*x2[i]*(126233085.0*z16[i] - 463991880.0*z14[i]*r2[i] + 695987820.0*z12[i]*r4[i] - 548354040.0*z10[i]*r6[i] + 243221550.0*z8[i]*r8[i] - 60386040.0*z6[i]*r10[i] + 7827820.0*z4[i]*r12[i] - 447304.0*z2[i]*r14[i] + 7293.0*r16[i]) + 1641030105.0*z18[i] - 7195285845.0*z16[i]*r2[i] + 13223768580.0*z14[i]*r4[i] - 13223768580.0*z12[i]*r6[i] + 7814045070.0*z10[i]*r8[i] - 2772725670.0*z8[i]*r10[i] + 573667380.0*z6[i]*r12[i] - 63740820.0*z4[i]*r14[i] + 3187041.0*z2[i]*r16[i] - 46189.0*r18[i]) ;
prCofDX[totalAN*418+i] = 0.00408295749053327*x[i]*(149184555.0*z18[i] - 585262485.0*z16[i]*r2[i] + 949074300.0*z14[i]*r4[i] - 822531060.0*z12[i]*r6[i] + 411265530.0*z10[i]*r8[i] - 119399670.0*z8[i]*r10[i] + 19213740.0*z6[i]*r12[i] - 1524900.0*z4[i]*r14[i] + 45747.0*z2[i]*r16[i] - 51.0*(x2[i] - y2[i])*(11475735.0*z16[i] - 37218600.0*z14[i]*r2[i] + 48384180.0*z12[i]*r4[i] - 32256120.0*z10[i]*r6[i] + 11705850.0*z8[i]*r8[i] - 2260440.0*z6[i]*r10[i] + 209300.0*z4[i]*r12[i] - 7176.0*z2[i]*r14[i] + 39.0*r16[i]) - 221.0*r18[i]) ;
prCofDX[totalAN*419+i] = 0.0138459825039349*z[i]*(-272.0*x2[i]*(x2[i] - 3.0*y2[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]) + 3.0*(x2[i] - y2[i])*(19458855.0*z16[i] - 67856520.0*z14[i]*r2[i] + 96282900.0*z12[i]*r4[i] - 71524440.0*z10[i]*r6[i] + 29801850.0*z8[i]*r8[i] - 6921720.0*z6[i]*r10[i] + 835380.0*z4[i]*r12[i] - 44200.0*z2[i]*r14[i] + 663.0*r16[i])) ;
prCofDX[totalAN*420+i] = 0.139837568674965*x[i]*((x2[i] - 3.0*y2[i])*(6486285.0*z16[i] - 19957800.0*z14[i]*r2[i] + 24542700.0*z12[i]*r4[i] - 15426840.0*z10[i]*r6[i] + 5259150.0*z8[i]*r8[i] - 950040.0*z6[i]*r10[i] + 81900.0*z4[i]*r12[i] - 2600.0*z2[i]*r14[i] + 13.0*r16[i]) - 4.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i])) ;
prCofDX[totalAN*421+i] = 0.699187843374825*z[i]*(-2.0*x2[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i]) + (x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(1297257.0*z14[i] - 3492615.0*z12[i]*r2[i] + 3681405.0*z10[i]*r4[i] - 1928355.0*z8[i]*r6[i] + 525915.0*z6[i]*r8[i] - 71253.0*z4[i]*r10[i] + 4095.0*z2[i]*r12[i] - 65.0*r14[i])) ;
prCofDX[totalAN*422+i] = 0.920523570163618*x[i]*(3.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(299367.0*z14[i] - 698523.0*z12[i]*r2[i] + 623007.0*z10[i]*r4[i] - 267003.0*z8[i]*r6[i] + 56637.0*z6[i]*r8[i] - 5481.0*z4[i]*r10[i] + 189.0*z2[i]*r12[i] - r14[i]) - (x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i])) ;
prCofDX[totalAN*423+i] = 0.426119611785933*z[i]*(-4.0*x2[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i]) + 7.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(232841.0*z12[i] - 465682.0*z10[i]*r2[i] + 346115.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 18879.0*z4[i]*r8[i] - 1218.0*z2[i]*r10[i] + 21.0*r12[i])) ;
prCofDX[totalAN*424+i] = 0.625372618731034*x[i]*(2.0*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(432419.0*z12[i] - 731786.0*z10[i]*r2[i] + 445005.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 13485.0*z4[i]*r8[i] - 522.0*z2[i]*r10[i] + 3.0*r12[i]) - (x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i])) ;
prCofDX[totalAN*425+i] = 0.972181244054524*z[i]*(-2.0*x2[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i]) + 9.0*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(44733.0*z10[i] - 63085.0*z8[i]*r2[i] + 30690.0*z6[i]*r4[i] - 6138.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 9.0*r10[i])) ;
prCofDX[totalAN*426+i] = 1.60550448425925*x[i]*((x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(164021.0*z10[i] - 189255.0*z8[i]*r2[i] + 71610.0*z6[i]*r4[i] - 10230.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 3.0*r10[i]) - 3.0*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i])) ;
prCofDX[totalAN*427+i] = 2.82678234249248*z[i]*(-24.0*x2[i]*(407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) + 11.0*(5291.0*z8[i] - 4884.0*z6[i]*r2[i] + 1386.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + 3.0*r8[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i])) ;
prCofDX[totalAN*428+i] = 1.99883696331483*x[i]*(-2.0*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]) + 3.0*(15873.0*z8[i] - 11396.0*z6[i]*r2[i] + 2310.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + r8[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i])) ;
prCofDX[totalAN*429+i] = 8.1193141272878*z[i]*(-2.0*x2[i]*(259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]) + 13.0*(481.0*z6[i] - 259.0*z4[i]*r2[i] + 35.0*z2[i]*r4[i] - r6[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i])) ;
prCofDX[totalAN*430+i] = 1.05259392999953*x[i]*(-(1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]) + 7.0*(3367.0*z6[i] - 1295.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - r6[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i])) ;
prCofDX[totalAN*431+i] = 2.54225586672922*z[i]*(-4.0*x2[i]*(-3.0*x2[i] - 3.0*y2[i] + 34.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]) + 3.0*(1443.0*z4[i] - 370.0*z2[i]*r2[i] + 15.0*r4[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i])) ;
prCofDX[totalAN*432+i] = 2.27386277368167*x[i]*(-(-x2[i] - y2[i] + 36.0*z2[i])*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]) + 4.0*(481.0*z4[i] - 74.0*z2[i]*r2[i] + r4[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i])) ;
prCofDX[totalAN*433+i] = 6.91568363939543*z[i]*(-2.0*x2[i]*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]) + 17.0*(-x2[i] - y2[i] + 12.0*z2[i])*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i])) ;
prCofDX[totalAN*434+i] = 1.29542623480908*x[i]*(-x18[i] + 153.0*x16[i]*y2[i] - 3060.0*x14[i]*y4[i] + 18564.0*x12[i]*y6[i] - 43758.0*x10[i]*y8[i] + 43758.0*x8[i]*y10[i] - 18564.0*x6[i]*y12[i] + 3060.0*x4[i]*y14[i] - 153.0*x2[i]*y16[i] + y18[i] - 9.0*(x2[i] + y2[i] - 38.0*z2[i])*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i])) ;
prCofDX[totalAN*435+i] = 108.688499895112*z[i]*(x18[i] - 153.0*x16[i]*y2[i] + 3060.0*x14[i]*y4[i] - 18564.0*x12[i]*y6[i] + 43758.0*x10[i]*y8[i] - 43758.0*x8[i]*y10[i] + 18564.0*x6[i]*y12[i] - 3060.0*x4[i]*y14[i] + 153.0*x2[i]*y16[i] - y18[i]) ;
prCofDX[totalAN*436+i] = 18.0896429018698*x[i]*(x18[i] - 171.0*x16[i]*y2[i] + 3876.0*x14[i]*y4[i] - 27132.0*x12[i]*y6[i] + 75582.0*x10[i]*y8[i] - 92378.0*x8[i]*y10[i] + 50388.0*x6[i]*y12[i] - 11628.0*x4[i]*y14[i] + 969.0*x2[i]*y16[i] - 19.0*y18[i]) ;

prCofDY[totalAN*396+i] = 18.0896429018698*x[i]*(x18[i] - 171.0*x16[i]*y2[i] + 3876.0*x14[i]*y4[i] - 27132.0*x12[i]*y6[i] + 75582.0*x10[i]*y8[i] - 92378.0*x8[i]*y10[i] + 50388.0*x6[i]*y12[i] - 11628.0*x4[i]*y14[i] + 969.0*x2[i]*y16[i] - 19.0*y18[i]) ;
prCofDY[totalAN*397+i] = 108.688499895112*z[i]*(x18[i] - 153.0*x16[i]*y2[i] + 3060.0*x14[i]*y4[i] - 18564.0*x12[i]*y6[i] + 43758.0*x10[i]*y8[i] - 43758.0*x8[i]*y10[i] + 18564.0*x6[i]*y12[i] - 3060.0*x4[i]*y14[i] + 153.0*x2[i]*y16[i] - y18[i]) ;
prCofDY[totalAN*398+i] = 1.29542623480908*x[i]*(y2[i]*(-18.0*x16[i] + 816.0*x14[i]*y2[i] - 8568.0*x12[i]*y4[i] + 31824.0*x10[i]*y6[i] - 48620.0*x8[i]*y8[i] + 31824.0*x6[i]*y10[i] - 8568.0*x4[i]*y12[i] + 816.0*x2[i]*y14[i] - 18.0*y16[i]) - 9.0*(x2[i] + y2[i] - 38.0*z2[i])*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i])) ;
prCofDY[totalAN*399+i] = 6.91568363939543*z[i]*(-2.0*y2[i]*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]) + 17.0*(-x2[i] - y2[i] + 12.0*z2[i])*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i])) ;
prCofDY[totalAN*400+i] = 9.09545109472669*x[i]*(-4.0*y2[i]*(-x2[i] - y2[i] + 36.0*z2[i])*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]) + (481.0*z4[i] - 74.0*z2[i]*r2[i] + r4[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i])) ;
prCofDY[totalAN*401+i] = 2.54225586672922*z[i]*(-4.0*y2[i]*(-3.0*x2[i] - 3.0*y2[i] + 34.0*z2[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]) + 3.0*(1443.0*z4[i] - 370.0*z2[i]*r2[i] + 15.0*r4[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i])) ;
prCofDY[totalAN*402+i] = 1.05259392999953*x[i]*(-2.0*y2[i]*(1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]) + 7.0*(3367.0*z6[i] - 1295.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - r6[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i])) ;
prCofDY[totalAN*403+i] = 8.1193141272878*z[i]*(-2.0*y2[i]*(259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]) + 13.0*(481.0*z6[i] - 259.0*z4[i]*r2[i] + 35.0*z2[i]*r4[i] - r6[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i])) ;
prCofDY[totalAN*404+i] = 1.99883696331483*x[i]*(-8.0*y2[i]*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]) + 3.0*(15873.0*z8[i] - 11396.0*z6[i]*r2[i] + 2310.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + r8[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i])) ;
prCofDY[totalAN*405+i] = 2.82678234249248*z[i]*(-24.0*y2[i]*(407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) + 11.0*(5291.0*z8[i] - 4884.0*z6[i]*r2[i] + 1386.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + 3.0*r8[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i])) ;
prCofDY[totalAN*406+i] = 1.60550448425925*x[i]*(-6.0*y2[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i])*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i]) + (x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(164021.0*z10[i] - 189255.0*z8[i]*r2[i] + 71610.0*z6[i]*r4[i] - 10230.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 3.0*r10[i])) ;
prCofDY[totalAN*407+i] = 0.972181244054524*z[i]*(-2.0*y2[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i]) + 9.0*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(44733.0*z10[i] - 63085.0*z8[i]*r2[i] + 30690.0*z6[i]*r4[i] - 6138.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 9.0*r10[i])) ;
prCofDY[totalAN*408+i] = 1.25074523746207*x[i]*(-4.0*y2[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i]) + (x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(432419.0*z12[i] - 731786.0*z10[i]*r2[i] + 445005.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 13485.0*z4[i]*r8[i] - 522.0*z2[i]*r10[i] + 3.0*r12[i])) ;
prCofDY[totalAN*409+i] = 0.426119611785933*z[i]*(-4.0*y2[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i]) + 7.0*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(232841.0*z12[i] - 465682.0*z10[i]*r2[i] + 346115.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 18879.0*z4[i]*r8[i] - 1218.0*z2[i]*r10[i] + 21.0*r12[i])) ;
prCofDY[totalAN*410+i] = 0.920523570163618*x[i]*(-2.0*y2[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i]) + 3.0*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(299367.0*z14[i] - 698523.0*z12[i]*r2[i] + 623007.0*z10[i]*r4[i] - 267003.0*z8[i]*r6[i] + 56637.0*z6[i]*r8[i] - 5481.0*z4[i]*r10[i] + 189.0*z2[i]*r12[i] - r14[i])) ;
prCofDY[totalAN*411+i] = 0.699187843374825*z[i]*(-2.0*y2[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i]) + (x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(1297257.0*z14[i] - 3492615.0*z12[i]*r2[i] + 3681405.0*z10[i]*r4[i] - 1928355.0*z8[i]*r6[i] + 525915.0*z6[i]*r8[i] - 71253.0*z4[i]*r10[i] + 4095.0*z2[i]*r12[i] - 65.0*r14[i])) ;
prCofDY[totalAN*412+i] = 0.139837568674965*x[i]*(-16.0*y2[i]*(x2[i] - y2[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i]) + (x2[i] - 3.0*y2[i])*(6486285.0*z16[i] - 19957800.0*z14[i]*r2[i] + 24542700.0*z12[i]*r4[i] - 15426840.0*z10[i]*r6[i] + 5259150.0*z8[i]*r8[i] - 950040.0*z6[i]*r10[i] + 81900.0*z4[i]*r12[i] - 2600.0*z2[i]*r14[i] + 13.0*r16[i])) ;
prCofDY[totalAN*413+i] = 0.0138459825039349*z[i]*(-272.0*y2[i]*(3.0*x2[i] - y2[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]) + 3.0*(x2[i] - y2[i])*(19458855.0*z16[i] - 67856520.0*z14[i]*r2[i] + 96282900.0*z12[i]*r4[i] - 71524440.0*z10[i]*r6[i] + 29801850.0*z8[i]*r8[i] - 6921720.0*z6[i]*r10[i] + 835380.0*z4[i]*r12[i] - 44200.0*z2[i]*r14[i] + 663.0*r16[i])) ;
prCofDY[totalAN*414+i] = 0.00408295749053327*x[i]*(-102.0*y2[i]*(11475735.0*z16[i] - 37218600.0*z14[i]*r2[i] + 48384180.0*z12[i]*r4[i] - 32256120.0*z10[i]*r6[i] + 11705850.0*z8[i]*r8[i] - 2260440.0*z6[i]*r10[i] + 209300.0*z4[i]*r12[i] - 7176.0*z2[i]*r14[i] + 39.0*r16[i]) + 149184555.0*z18[i] - 585262485.0*z16[i]*r2[i] + 949074300.0*z14[i]*r4[i] - 822531060.0*z12[i]*r6[i] + 411265530.0*z10[i]*r8[i] - 119399670.0*z8[i]*r10[i] + 19213740.0*z6[i]*r12[i] - 1524900.0*z4[i]*r14[i] + 45747.0*z2[i]*r16[i] - 221.0*r18[i]) ;
prCofDY[totalAN*415+i] = 0.000199703978712595*z[i]*(-114.0*y2[i]*(126233085.0*z16[i] - 463991880.0*z14[i]*r2[i] + 695987820.0*z12[i]*r4[i] - 548354040.0*z10[i]*r6[i] + 243221550.0*z8[i]*r8[i] - 60386040.0*z6[i]*r10[i] + 7827820.0*z4[i]*r12[i] - 447304.0*z2[i]*r14[i] + 7293.0*r16[i]) + 1641030105.0*z18[i] - 7195285845.0*z16[i]*r2[i] + 13223768580.0*z14[i]*r4[i] - 13223768580.0*z12[i]*r6[i] + 7814045070.0*z10[i]*r8[i] - 2772725670.0*z8[i]*r10[i] + 573667380.0*z6[i]*r12[i] - 63740820.0*z4[i]*r14[i] + 3187041.0*z2[i]*r16[i] - 46189.0*r18[i]) ;
prCofDY[totalAN*416+i] = 0.00261836791769081*y[i]*(-883631595.0*z18[i] + 3653936055.0*z16[i]*r2[i] - 6263890380.0*z14[i]*r4[i] + 5757717420.0*z12[i]*r6[i] - 3064591530.0*z10[i]*r8[i] + 951080130.0*z8[i]*r10[i] - 164384220.0*z6[i]*r12[i] + 14090076.0*z4[i]*r14[i] - 459459.0*z2[i]*r16[i] + 2431.0*r18[i]) ;
prCofDY[totalAN*417+i] = -0.0227662535732358*x[i]*y[i]*z[i]*(126233085.0*z16[i] - 463991880.0*z14[i]*r2[i] + 695987820.0*z12[i]*r4[i] - 548354040.0*z10[i]*r6[i] + 243221550.0*z8[i]*r8[i] - 60386040.0*z6[i]*r10[i] + 7827820.0*z4[i]*r12[i] - 447304.0*z2[i]*r14[i] + 7293.0*r16[i]) ;
prCofDY[totalAN*418+i] = 0.00408295749053327*y[i]*(-149184555.0*z18[i] + 585262485.0*z16[i]*r2[i] - 949074300.0*z14[i]*r4[i] + 822531060.0*z12[i]*r6[i] - 411265530.0*z10[i]*r8[i] + 119399670.0*z8[i]*r10[i] - 19213740.0*z6[i]*r12[i] + 1524900.0*z4[i]*r14[i] - 45747.0*z2[i]*r16[i] - 51.0*(x2[i] - y2[i])*(11475735.0*z16[i] - 37218600.0*z14[i]*r2[i] + 48384180.0*z12[i]*r4[i] - 32256120.0*z10[i]*r6[i] + 11705850.0*z8[i]*r8[i] - 2260440.0*z6[i]*r10[i] + 209300.0*z4[i]*r12[i] - 7176.0*z2[i]*r14[i] + 39.0*r16[i]) + 221.0*r18[i]) ;
prCofDY[totalAN*419+i] = 0.0276919650078697*x[i]*y[i]*z[i]*(-58376565.0*z16[i] + 203569560.0*z14[i]*r2[i] - 288848700.0*z12[i]*r4[i] + 214573320.0*z10[i]*r6[i] - 89405550.0*z8[i]*r8[i] + 20765160.0*z6[i]*r10[i] - 2506140.0*z4[i]*r12[i] + 132600.0*z2[i]*r14[i] - 136.0*(x2[i] - 3.0*y2[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]) - 1989.0*r16[i]) ;
prCofDY[totalAN*420+i] = -0.139837568674965*y[i]*((3.0*x2[i] - y2[i])*(6486285.0*z16[i] - 19957800.0*z14[i]*r2[i] + 24542700.0*z12[i]*r4[i] - 15426840.0*z10[i]*r6[i] + 5259150.0*z8[i]*r8[i] - 950040.0*z6[i]*r10[i] + 81900.0*z4[i]*r12[i] - 2600.0*z2[i]*r14[i] + 13.0*r16[i]) + 4.0*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i])) ;
prCofDY[totalAN*421+i] = -1.39837568674965*x[i]*y[i]*z[i]*(2.0*(x2[i] - y2[i])*(1297257.0*z14[i] - 3492615.0*z12[i]*r2[i] + 3681405.0*z10[i]*r4[i] - 1928355.0*z8[i]*r6[i] + 525915.0*z6[i]*r8[i] - 71253.0*z4[i]*r10[i] + 4095.0*z2[i]*r12[i] - 65.0*r14[i]) + (x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i])) ;
prCofDY[totalAN*422+i] = -0.920523570163618*y[i]*(3.0*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(299367.0*z14[i] - 698523.0*z12[i]*r2[i] + 623007.0*z10[i]*r4[i] - 267003.0*z8[i]*r6[i] + 56637.0*z6[i]*r8[i] - 5481.0*z4[i]*r10[i] + 189.0*z2[i]*r12[i] - r14[i]) + (x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i])) ;
prCofDY[totalAN*423+i] = -0.852239223571866*x[i]*y[i]*z[i]*(7.0*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(232841.0*z12[i] - 465682.0*z10[i]*r2[i] + 346115.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 18879.0*z4[i]*r8[i] - 1218.0*z2[i]*r10[i] + 21.0*r12[i]) + 2.0*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i])) ;
prCofDY[totalAN*424+i] = -0.625372618731034*y[i]*(2.0*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(432419.0*z12[i] - 731786.0*z10[i]*r2[i] + 445005.0*z8[i]*r4[i] - 118668.0*z6[i]*r6[i] + 13485.0*z4[i]*r8[i] - 522.0*z2[i]*r10[i] + 3.0*r12[i]) + (x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i])) ;
prCofDY[totalAN*425+i] = -1.94436248810905*x[i]*y[i]*z[i]*(36.0*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(44733.0*z10[i] - 63085.0*z8[i]*r2[i] + 30690.0*z6[i]*r4[i] - 6138.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 9.0*r10[i]) + (x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i])) ;
prCofDY[totalAN*426+i] = -1.60550448425925*y[i]*((9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(164021.0*z10[i] - 189255.0*z8[i]*r2[i] + 71610.0*z6[i]*r4[i] - 10230.0*z4[i]*r6[i] + 465.0*z2[i]*r8[i] - 3.0*r10[i]) + 3.0*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i])) ;
prCofDY[totalAN*427+i] = -5.65356468498497*x[i]*y[i]*z[i]*(12.0*(407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) + 11.0*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i])*(5291.0*z8[i] - 4884.0*z6[i]*r2[i] + 1386.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + 3.0*r8[i])) ;
prCofDY[totalAN*428+i] = -1.99883696331483*y[i]*(2.0*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]) + 3.0*(15873.0*z8[i] - 11396.0*z6[i]*r2[i] + 2310.0*z4[i]*r4[i] - 132.0*z2[i]*r6[i] + r8[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i])) ;
prCofDY[totalAN*429+i] = -16.2386282545756*x[i]*y[i]*z[i]*((259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]) + 26.0*(481.0*z6[i] - 259.0*z4[i]*r2[i] + 35.0*z2[i]*r4[i] - r6[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i])) ;
prCofDY[totalAN*430+i] = -1.05259392999953*y[i]*((1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]) + 7.0*(3367.0*z6[i] - 1295.0*z4[i]*r2[i] + 105.0*z2[i]*r4[i] - r6[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i])) ;
prCofDY[totalAN*431+i] = 5.08451173345844*x[i]*y[i]*z[i]*(2.0*(3.0*x2[i] + 3.0*y2[i] - 34.0*z2[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]) - 3.0*(1443.0*z4[i] - 370.0*z2[i]*r2[i] + 15.0*r4[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i])) ;
prCofDY[totalAN*432+i] = 2.27386277368167*y[i]*((x2[i] + y2[i] - 36.0*z2[i])*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]) - 4.0*(481.0*z4[i] - 74.0*z2[i]*r2[i] + r4[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i])) ;
prCofDY[totalAN*433+i] = 13.8313672787909*x[i]*y[i]*z[i]*(-x16[i] + 136.0*x14[i]*y2[i] - 2380.0*x12[i]*y4[i] + 12376.0*x10[i]*y6[i] - 24310.0*x8[i]*y8[i] + 19448.0*x6[i]*y10[i] - 6188.0*x4[i]*y12[i] + 680.0*x2[i]*y14[i] - 17.0*y16[i] - 136.0*(-x2[i] - y2[i] + 12.0*z2[i])*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i])) ;
prCofDY[totalAN*434+i] = 1.29542623480908*y[i]*(-x18[i] + 153.0*x16[i]*y2[i] - 3060.0*x14[i]*y4[i] + 18564.0*x12[i]*y6[i] - 43758.0*x10[i]*y8[i] + 43758.0*x8[i]*y10[i] - 18564.0*x6[i]*y12[i] + 3060.0*x4[i]*y14[i] - 153.0*x2[i]*y16[i] + y18[i] + 9.0*(x2[i] + y2[i] - 38.0*z2[i])*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i])) ;
prCofDY[totalAN*435+i] = 217.376999790224*x[i]*y[i]*z[i]*(-9.0*x16[i] + 408.0*x14[i]*y2[i] - 4284.0*x12[i]*y4[i] + 15912.0*x10[i]*y6[i] - 24310.0*x8[i]*y8[i] + 15912.0*x6[i]*y10[i] - 4284.0*x4[i]*y12[i] + 408.0*x2[i]*y14[i] - 9.0*y16[i]) ;
prCofDY[totalAN*436+i] = 18.0896429018698*y[i]*(-19.0*x18[i] + 969.0*x16[i]*y2[i] - 11628.0*x14[i]*y4[i] + 50388.0*x12[i]*y6[i] - 92378.0*x10[i]*y8[i] + 75582.0*x8[i]*y10[i] - 27132.0*x6[i]*y12[i] + 3876.0*x4[i]*y14[i] - 171.0*x2[i]*y16[i] + y18[i]) ;

prCofDZ[totalAN*396+i] = 0 ;
prCofDZ[totalAN*397+i] = 5.72044736290064*y[i]*(19.0*x18[i] - 969.0*x16[i]*y2[i] + 11628.0*x14[i]*y4[i] - 50388.0*x12[i]*y6[i] + 92378.0*x10[i]*y8[i] - 75582.0*x8[i]*y10[i] + 27132.0*x6[i]*y12[i] - 3876.0*x4[i]*y14[i] + 171.0*x2[i]*y16[i] - y18[i]) ;
prCofDZ[totalAN*398+i] = 98.4523938454903*x[i]*y[i]*z[i]*(9.0*x16[i] - 408.0*x14[i]*y2[i] + 4284.0*x12[i]*y4[i] - 15912.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 15912.0*x6[i]*y10[i] + 4284.0*x4[i]*y12[i] - 408.0*x2[i]*y14[i] + 9.0*y16[i]) ;
prCofDZ[totalAN*399+i] = -6.91568363939543*y[i]*(x2[i] + y2[i] - 36.0*z2[i])*(17.0*x16[i] - 680.0*x14[i]*y2[i] + 6188.0*x12[i]*y4[i] - 19448.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 12376.0*x6[i]*y10[i] + 2380.0*x4[i]*y12[i] - 136.0*x2[i]*y14[i] + y16[i]) ;
prCofDZ[totalAN*400+i] = 436.581652546881*x[i]*y[i]*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 34.0*z2[i])*(x14[i] - 35.0*x12[i]*y2[i] + 273.0*x10[i]*y4[i] - 715.0*x8[i]*y6[i] + 715.0*x6[i]*y8[i] - 273.0*x4[i]*y10[i] + 35.0*x2[i]*y12[i] - y14[i]) ;
prCofDZ[totalAN*401+i] = 2.54225586672922*y[i]*(1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(15.0*x14[i] - 455.0*x12[i]*y2[i] + 3003.0*x10[i]*y4[i] - 6435.0*x8[i]*y6[i] + 5005.0*x6[i]*y8[i] - 1365.0*x4[i]*y10[i] + 105.0*x2[i]*y12[i] - y14[i]) ;
prCofDZ[totalAN*402+i] = 71.576387239968*x[i]*y[i]*z[i]*(259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(7.0*x12[i] - 182.0*x10[i]*y2[i] + 1001.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1001.0*x4[i]*y8[i] - 182.0*x2[i]*y10[i] + 7.0*y12[i]) ;
prCofDZ[totalAN*403+i] = 8.1193141272878*y[i]*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(13.0*x12[i] - 286.0*x10[i]*y2[i] + 1287.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 715.0*x4[i]*y8[i] - 78.0*x2[i]*y10[i] + y12[i]) ;
prCofDZ[totalAN*404+i] = 511.702262608596*x[i]*y[i]*z[i]*(407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(3.0*x10[i] - 55.0*x8[i]*y2[i] + 198.0*x6[i]*y4[i] - 198.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - 3.0*y10[i]) ;
prCofDZ[totalAN*405+i] = 8.48034702747746*y[i]*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i])*(11.0*x10[i] - 165.0*x8[i]*y2[i] + 462.0*x6[i]*y4[i] - 330.0*x4[i]*y6[i] + 55.0*x2[i]*y8[i] - y10[i]) ;
prCofDZ[totalAN*406+i] = 6.42201793703699*x[i]*y[i]*z[i]*(5.0*x8[i] - 60.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 60.0*x2[i]*y6[i] + 5.0*y8[i])*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i]) ;
prCofDZ[totalAN*407+i] = 0.972181244054524*y[i]*(9.0*x8[i] - 84.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 36.0*x2[i]*y6[i] + y8[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i]) ;
prCofDZ[totalAN*408+i] = 20.0119237993931*x[i]*y[i]*z[i]*(x6[i] - 7.0*x4[i]*y2[i] + 7.0*x2[i]*y4[i] - y6[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i]) ;
prCofDZ[totalAN*409+i] = 1.2783588353578*y[i]*(7.0*x6[i] - 35.0*x4[i]*y2[i] + 21.0*x2[i]*y4[i] - y6[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i]) ;
prCofDZ[totalAN*410+i] = 3.68209428065447*x[i]*y[i]*z[i]*(3.0*x4[i] - 10.0*x2[i]*y2[i] + 3.0*y4[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i]) ;
prCofDZ[totalAN*411+i] = 0.699187843374825*y[i]*(5.0*x4[i] - 10.0*x2[i]*y2[i] + y4[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i]) ;
prCofDZ[totalAN*412+i] = 17.8992087903955*x[i]*y[i]*z[i]*(x2[i] - y2[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]) ;
prCofDZ[totalAN*413+i] = 0.235381702566892*y[i]*(3.0*x2[i] - y2[i])*(11475735.0*z16[i] - 37218600.0*z14[i]*r2[i] + 48384180.0*z12[i]*r4[i] - 32256120.0*z10[i]*r6[i] + 11705850.0*z8[i]*r8[i] - 2260440.0*z6[i]*r10[i] + 209300.0*z4[i]*r12[i] - 7176.0*z2[i]*r14[i] + 39.0*r16[i]) ;
prCofDZ[totalAN*414+i] = 0.0489954898863992*x[i]*y[i]*z[i]*(126233085.0*z16[i] - 463991880.0*z14[i]*r2[i] + 695987820.0*z12[i]*r4[i] - 548354040.0*z10[i]*r6[i] + 243221550.0*z8[i]*r8[i] - 60386040.0*z6[i]*r10[i] + 7827820.0*z4[i]*r12[i] - 447304.0*z2[i]*r14[i] + 7293.0*r16[i]) ;
prCofDZ[totalAN*415+i] = 0.0037943755955393*y[i]*(883631595.0*z18[i] - 3653936055.0*z16[i]*r2[i] + 6263890380.0*z14[i]*r4[i] - 5757717420.0*z12[i]*r6[i] + 3064591530.0*z10[i]*r8[i] - 951080130.0*z8[i]*r10[i] + 164384220.0*z6[i]*r12[i] - 14090076.0*z4[i]*r14[i] + 459459.0*z2[i]*r16[i] - 2431.0*r18[i]) ;
prCofDZ[totalAN*416+i] = 0.000551235351092801*z[i]*(4418157975.0*z18[i] - 20419054425.0*z16[i]*r2[i] + 39671305740.0*z14[i]*r4[i] - 42075627300.0*z12[i]*r6[i] + 26466926850.0*z10[i]*r8[i] - 10039179150.0*z8[i]*r10[i] + 2230928700.0*z6[i]*r12[i] - 267711444.0*z4[i]*r14[i] + 14549535.0*z2[i]*r16[i] - 230945.0*r18[i]) ;
prCofDZ[totalAN*417+i] = 0.0037943755955393*x[i]*(883631595.0*z18[i] - 3653936055.0*z16[i]*r2[i] + 6263890380.0*z14[i]*r4[i] - 5757717420.0*z12[i]*r6[i] + 3064591530.0*z10[i]*r8[i] - 951080130.0*z8[i]*r10[i] + 164384220.0*z6[i]*r12[i] - 14090076.0*z4[i]*r14[i] + 459459.0*z2[i]*r16[i] - 2431.0*r18[i]) ;
prCofDZ[totalAN*418+i] = 0.0244977449431996*z[i]*(x2[i] - y2[i])*(126233085.0*z16[i] - 463991880.0*z14[i]*r2[i] + 695987820.0*z12[i]*r4[i] - 548354040.0*z10[i]*r6[i] + 243221550.0*z8[i]*r8[i] - 60386040.0*z6[i]*r10[i] + 7827820.0*z4[i]*r12[i] - 447304.0*z2[i]*r14[i] + 7293.0*r16[i]) ;
prCofDZ[totalAN*419+i] = 0.235381702566892*x[i]*(x2[i] - 3.0*y2[i])*(11475735.0*z16[i] - 37218600.0*z14[i]*r2[i] + 48384180.0*z12[i]*r4[i] - 32256120.0*z10[i]*r6[i] + 11705850.0*z8[i]*r8[i] - 2260440.0*z6[i]*r10[i] + 209300.0*z4[i]*r12[i] - 7176.0*z2[i]*r14[i] + 39.0*r16[i]) ;
prCofDZ[totalAN*420+i] = 4.47480219759888*z[i]*(x4[i] - 6.0*x2[i]*y2[i] + y4[i])*(498945.0*z14[i] - 1415925.0*z12[i]*r2[i] + 1577745.0*z10[i]*r4[i] - 876525.0*z8[i]*r6[i] + 254475.0*z6[i]*r8[i] - 36855.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 39.0*r14[i]) ;
prCofDZ[totalAN*421+i] = 0.699187843374825*x[i]*(x4[i] - 10.0*x2[i]*y2[i] + 5.0*y4[i])*(2494725.0*z14[i] - 6135675.0*z12[i]*r2[i] + 5785065.0*z10[i]*r4[i] - 2629575.0*z8[i]*r6[i] + 593775.0*z6[i]*r8[i] - 61425.0*z4[i]*r10[i] + 2275.0*z2[i]*r12[i] - 13.0*r14[i]) ;
prCofDZ[totalAN*422+i] = 1.84104714032724*z[i]*(x6[i] - 15.0*x4[i]*y2[i] + 15.0*x2[i]*y4[i] - y6[i])*(698523.0*z12[i] - 1472562.0*z10[i]*r2[i] + 1157013.0*z8[i]*r4[i] - 420732.0*z6[i]*r6[i] + 71253.0*z4[i]*r8[i] - 4914.0*z2[i]*r10[i] + 91.0*r12[i]) ;
prCofDZ[totalAN*423+i] = 1.2783588353578*x[i]*(x6[i] - 21.0*x4[i]*y2[i] + 35.0*x2[i]*y4[i] - 7.0*y6[i])*(698523.0*z12[i] - 1246014.0*z10[i]*r2[i] + 801009.0*z8[i]*r4[i] - 226548.0*z6[i]*r6[i] + 27405.0*z4[i]*r8[i] - 1134.0*z2[i]*r10[i] + 7.0*r12[i]) ;
prCofDZ[totalAN*424+i] = 2.50149047492414*z[i]*(x8[i] - 28.0*x6[i]*y2[i] + 70.0*x4[i]*y4[i] - 28.0*x2[i]*y6[i] + y8[i])*(232841.0*z10[i] - 346115.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 37758.0*z4[i]*r6[i] + 3045.0*z2[i]*r8[i] - 63.0*r10[i]) ;
prCofDZ[totalAN*425+i] = 0.972181244054524*x[i]*(x8[i] - 36.0*x6[i]*y2[i] + 126.0*x4[i]*y4[i] - 84.0*x2[i]*y6[i] + 9.0*y8[i])*(365893.0*z10[i] - 445005.0*z8[i]*r2[i] + 178002.0*z6[i]*r4[i] - 26970.0*z4[i]*r6[i] + 1305.0*z2[i]*r8[i] - 9.0*r10[i]) ;
prCofDZ[totalAN*426+i] = 3.21100896851849*z[i]*(63085.0*z8[i] - 61380.0*z6[i]*r2[i] + 18414.0*z4[i]*r4[i] - 1860.0*z2[i]*r6[i] + 45.0*r8[i])*(x10[i] - 45.0*x8[i]*y2[i] + 210.0*x6[i]*y4[i] - 210.0*x4[i]*y6[i] + 45.0*x2[i]*y8[i] - y10[i]) ;
prCofDZ[totalAN*427+i] = 8.48034702747746*x[i]*(12617.0*z8[i] - 9548.0*z6[i]*r2[i] + 2046.0*z4[i]*r4[i] - 124.0*z2[i]*r6[i] + r8[i])*(x10[i] - 55.0*x8[i]*y2[i] + 330.0*x6[i]*y4[i] - 462.0*x4[i]*y6[i] + 165.0*x2[i]*y8[i] - 11.0*y10[i]) ;
prCofDZ[totalAN*428+i] = 127.925565652149*z[i]*(407.0*z6[i] - 231.0*z4[i]*r2[i] + 33.0*z2[i]*r4[i] - r6[i])*(x12[i] - 66.0*x10[i]*y2[i] + 495.0*x8[i]*y4[i] - 924.0*x6[i]*y6[i] + 495.0*x4[i]*y8[i] - 66.0*x2[i]*y10[i] + y12[i]) ;
prCofDZ[totalAN*429+i] = 8.1193141272878*x[i]*(2849.0*z6[i] - 1155.0*z4[i]*r2[i] + 99.0*z2[i]*r4[i] - r6[i])*(x12[i] - 78.0*x10[i]*y2[i] + 715.0*x8[i]*y4[i] - 1716.0*x6[i]*y6[i] + 1287.0*x4[i]*y8[i] - 286.0*x2[i]*y10[i] + 13.0*y12[i]) ;
prCofDZ[totalAN*430+i] = 35.788193619984*z[i]*(259.0*z4[i] - 70.0*z2[i]*r2[i] + 3.0*r4[i])*(x14[i] - 91.0*x12[i]*y2[i] + 1001.0*x10[i]*y4[i] - 3003.0*x8[i]*y6[i] + 3003.0*x6[i]*y8[i] - 1001.0*x4[i]*y10[i] + 91.0*x2[i]*y12[i] - y14[i]) ;
prCofDZ[totalAN*431+i] = 2.54225586672922*x[i]*(1295.0*z4[i] - 210.0*z2[i]*r2[i] + 3.0*r4[i])*(x14[i] - 105.0*x12[i]*y2[i] + 1365.0*x10[i]*y4[i] - 5005.0*x8[i]*y6[i] + 6435.0*x6[i]*y8[i] - 3003.0*x4[i]*y10[i] + 455.0*x2[i]*y12[i] - 15.0*y14[i]) ;
prCofDZ[totalAN*432+i] = 27.2863532841801*z[i]*(-3.0*x2[i] - 3.0*y2[i] + 34.0*z2[i])*(x16[i] - 120.0*x14[i]*y2[i] + 1820.0*x12[i]*y4[i] - 8008.0*x10[i]*y6[i] + 12870.0*x8[i]*y8[i] - 8008.0*x6[i]*y10[i] + 1820.0*x4[i]*y12[i] - 120.0*x2[i]*y14[i] + y16[i]) ;
prCofDZ[totalAN*433+i] = -6.91568363939543*x[i]*(x2[i] + y2[i] - 36.0*z2[i])*(x16[i] - 136.0*x14[i]*y2[i] + 2380.0*x12[i]*y4[i] - 12376.0*x10[i]*y6[i] + 24310.0*x8[i]*y8[i] - 19448.0*x6[i]*y10[i] + 6188.0*x4[i]*y12[i] - 680.0*x2[i]*y14[i] + 17.0*y16[i]) ;
prCofDZ[totalAN*434+i] = 49.2261969227451*z[i]*(x18[i] - 153.0*x16[i]*y2[i] + 3060.0*x14[i]*y4[i] - 18564.0*x12[i]*y6[i] + 43758.0*x10[i]*y8[i] - 43758.0*x8[i]*y10[i] + 18564.0*x6[i]*y12[i] - 3060.0*x4[i]*y14[i] + 153.0*x2[i]*y16[i] - y18[i]) ;
prCofDZ[totalAN*435+i] = 5.72044736290064*x[i]*(x18[i] - 171.0*x16[i]*y2[i] + 3876.0*x14[i]*y4[i] - 27132.0*x12[i]*y6[i] + 75582.0*x10[i]*y8[i] - 92378.0*x8[i]*y10[i] + 50388.0*x6[i]*y12[i] - 11628.0*x4[i]*y14[i] + 969.0*x2[i]*y16[i] - 19.0*y18[i]) ;
prCofDZ[totalAN*436+i] = 0 ;

        }
      

}}}}}}}}}}}}}}}}}
    }
  }
}}
//==============================================================================================================================
void getCD(
    py::detail::unchecked_mutable_reference<double, 5> &CDevX_mu,
    py::detail::unchecked_mutable_reference<double, 5> &CDevY_mu,
    py::detail::unchecked_mutable_reference<double, 5> &CDevZ_mu,
    double* prCofDX,
    double* prCofDY,
    double* prCofDZ,
    py::detail::unchecked_mutable_reference<double, 4> &C_mu,
    double* preCoef,
    double* x,
    double* y,
    double* z,
    double* r2,
    double* weights,
    double* bOa,
    double* aOa,
    double* exes,
    int totalAN,
    int Asize,
    int Ns,
    int Ntypes,
    int lMax,
    int posI,
    int posAtomI,
    int typeJ,
    const vector<int> &indices,
    bool attach,
    bool return_derivatives) {
  if (Asize == 0) {
    return;
  }
  double sumMe = 0; int NsNs = Ns*Ns; int LNsNs;
  int LNs;
  double preExp;
  double preVal;
  double preValX;
  double preValY;
  double preValZ;

  double  preVal1;
  double  preVal2;
  double  preVal3;
    
  double  preValX1;
  double  preValY1;
  double  preValZ1;

  double  preValX2;
  double  preValY2;
  double  preValZ2;

  double  preValX3;
  double  preValY3;
  double  preValZ3;
  double* preExponentArray = (double*) malloc(Ns*Asize*sizeof(double));

  // l=0-------------------------------------------------------------------------------------------------
  int shift = 0;
  for (int k = 0; k < Ns; k++) {
    for (int i = 0; i < Asize; i++) {
      preExponentArray[shift] = weights[i]*1.5707963267948966*exp(aOa[k]*r2[i]);
      shift++;
    }
  }

  shift = 0;
  for (int k = 0; k < Ns; k++) {
    sumMe = 0;
    for (int i = 0; i < Asize; i++) {
      preExp = preExponentArray[shift];
      sumMe += preExp;
      shift++;
    }
    for (int n = 0; n < Ns; n++) {
      C_mu(posI, typeJ, n, 0) += bOa[n*Ns + k]*sumMe;
    } 
  }

  shift = 0;
  if (return_derivatives) {
    for (int k = 0; k < Ns; k++) {
      for (int i = 0; i < Asize; i++) {
        preExp = preExponentArray[shift];
        shift++;
        preVal = 2.0*aOa[k]*preExp;
        preValX = preVal*x[i];
        preValY = preVal*y[i];
        preValZ = preVal*z[i];
        for (int n = 0; n < Ns; n++) {
          CDevX_mu(indices[i], posI, typeJ, n, 0) += bOa[n*Ns + k]*preValX;
          CDevY_mu(indices[i], posI, typeJ, n, 0) += bOa[n*Ns + k]*preValY;
          CDevZ_mu(indices[i], posI, typeJ, n, 0) += bOa[n*Ns + k]*preValZ;
        }
      } 
    }
  }

  // l=1-------------------------------------------------------------------------------------------------
  if (lMax > 0) { 
    LNsNs = NsNs;
    LNs = Ns;

    shift = 0 ;
    for (int k = 0; k < Ns; k++) {
      for (int i = 0; i < Asize; i++) {
        preExponentArray[shift] = weights[i]*2.7206990463849543*exp(aOa[LNs + k]*r2[i]);
        shift++;
      }
    }
    
    shift = 0 ;
    double sumMe1;
    double sumMe2;
    double sumMe3;
    for (int k = 0; k < Ns; k++) {
    sumMe1 = 0;
    sumMe2 = 0;
    sumMe3 = 0;
      for (int i = 0; i < Asize; i++) {
        preExp = preExponentArray[shift];
        sumMe1 += preExp*z[i];
        sumMe2 += preExp*x[i];
        sumMe3 += preExp*y[i];
        shift++;
      }
      for (int n = 0; n < Ns; n++) {
        C_mu(posI, typeJ, n, 1) += bOa[LNsNs + n*Ns + k]*sumMe1;
        C_mu(posI, typeJ, n, 2) += bOa[LNsNs + n*Ns + k]*sumMe2;
        C_mu(posI, typeJ, n, 3) += bOa[LNsNs + n*Ns + k]*sumMe3;
      }
    }

    if (return_derivatives) {
      shift = 0;
      for (int k = 0; k < Ns; k++) {
        for (int i = 0; i < Asize; i++) {
          preExp = preExponentArray[shift];
          shift++;
          preVal = 2.0*aOa[LNs + k]*preExp;
          if (return_derivatives) {
            preVal1 = preVal*z[i];
            preVal2 = preVal*x[i];
            preVal3 = preVal*y[i];
           
            preValX1 = preVal1*x[i];
            preValY1 = preVal1*y[i];
            preValZ1 = preVal1*z[i] + preExp;

            preValX2 = preVal2*x[i] + preExp;
            preValY2 = preVal2*y[i];
            preValZ2 = preVal2*z[i];

            preValX3 = preVal3*x[i];
            preValY3 = preVal3*y[i] + preExp;
            preValZ3 = preVal3*z[i];
          }
          for (int n = 0; n < Ns; n++) {
            CDevX_mu(indices[i], posI, typeJ, n, 1) += bOa[LNsNs + n*Ns + k]*preValX1;
            CDevY_mu(indices[i], posI, typeJ, n, 1) += bOa[LNsNs + n*Ns + k]*preValY1;
            CDevZ_mu(indices[i], posI, typeJ, n, 1) += bOa[LNsNs + n*Ns + k]*preValZ1;

            CDevX_mu(indices[i], posI, typeJ, n, 2) += bOa[LNsNs + n*Ns + k]*preValX2;
            CDevY_mu(indices[i], posI, typeJ, n, 2) += bOa[LNsNs + n*Ns + k]*preValY2;
            CDevZ_mu(indices[i], posI, typeJ, n, 2) += bOa[LNsNs + n*Ns + k]*preValZ2;

            CDevX_mu(indices[i], posI, typeJ, n, 3) += bOa[LNsNs + n*Ns + k]*preValX3;
            CDevY_mu(indices[i], posI, typeJ, n, 3) += bOa[LNsNs + n*Ns + k]*preValY3;
            CDevZ_mu(indices[i], posI, typeJ, n, 3) += bOa[LNsNs + n*Ns + k]*preValZ3;
          }
        }
      }
    }
  }

// l>2------------------------------------------------------------------------------------------------------
//
//[NsTs100*i_center*totalAN + Ns100*jd*totalAN + buffShift*totalAN*Ns + kd*totalAN + i_atom]
//
  if (lMax > 1) { 
    for (int restOfLs = 2; restOfLs <= lMax; restOfLs++) {	 
      LNsNs=restOfLs*NsNs; LNs=restOfLs*Ns; 
      shift = 0;
      for (int k = 0; k < Ns; k++) {
        for (int i = 0; i < Asize; i++) {
          double expSholder = aOa[LNs + k]*r2[i];
          preExponentArray[shift] = weights[i]*exp(expSholder);
          shift++;
        }
      }

      //double*  sumS = (double*) malloc(sizeof(double)*(restOfLs+1)*(restOfLs+1))
      shift = 0;
      for (int k = 0; k < Ns; k++) {
        for (int m = restOfLs*restOfLs; m < (restOfLs+1)*(restOfLs+1); m++) {
          sumMe = 0;
          for (int i = 0; i < Asize; i++) {
            preExp = preExponentArray[Asize*k+i];
            sumMe += preExp*preCoef[totalAN*(m-4)+i];
            shift++;
          }
          for (int n = 0; n < Ns; n++) {
            C_mu(posI, typeJ, n, m) += bOa[LNsNs + n*Ns + k]*sumMe;
          }
        }
      }

      if (return_derivatives) {
        shift = 0;
        for (int k = 0; k < Ns; k++) {
          for (int i = 0; i < Asize; i++) {
            preExp = preExponentArray[shift];
            shift++;
            for (int m = restOfLs*restOfLs; m < (restOfLs+1)*(restOfLs+1); m++) {
              preVal = 2.0*aOa[LNs + k]*preExp*preCoef[totalAN*(m-4)+i];
              preValX = x[i]*preVal + preExp*prCofDX[totalAN*(m-4)+i];
              preValY = y[i]*preVal + preExp*prCofDY[totalAN*(m-4)+i];
              preValZ = z[i]*preVal + preExp*prCofDZ[totalAN*(m-4)+i];
              for (int n = 0; n < Ns; n++) {
                CDevX_mu(indices[i], posI, typeJ, n, m) += bOa[LNsNs + n*Ns + k]*preValX;
                CDevY_mu(indices[i], posI, typeJ, n, m) += bOa[LNsNs + n*Ns + k]*preValY;
                CDevZ_mu(indices[i], posI, typeJ, n, m) += bOa[LNsNs + n*Ns + k]*preValZ;
              }
            }
          }
        }
      }
    }
  }
  free(preExponentArray);

  // If attach=True, the derivative with respect to the center atom coordinates
  // is the negative sum of derivatives with respect to coordinates of other
  // atoms in the the neighbourhood.
  if (return_derivatives && attach && (posAtomI >= 0)) {
    for (int m = 0; m < (lMax+1)*(lMax+1); m++) {
      for (int n = 0; n < Ns; n++) {
        double sumX = 0;
        double sumY = 0;
        double sumZ = 0;
        for (int i = 0; i < Asize; i++) {
          if (indices[i] != posAtomI) {
            sumX += CDevX_mu(indices[i], posI, typeJ,n,m);
            sumY += CDevY_mu(indices[i], posI, typeJ,n,m);
            sumZ += CDevZ_mu(indices[i], posI, typeJ,n,m);
          }
        }
        CDevX_mu(posAtomI, posI, typeJ,n,m) = -sumX;
        CDevY_mu(posAtomI, posI, typeJ,n,m) = -sumY;
        CDevZ_mu(posAtomI, posI, typeJ,n,m) = -sumZ;
      }
    }
  }
}
//================================================================================================
/**
 * Used to calculate the partial power spectrum.
 */
void getPD(
  py::detail::unchecked_mutable_reference<double, 2> &descriptor_mu,
  py::detail::unchecked_reference<double, 4> &Cnnd_u,
  int Ns,
  int Ts,
  int nCenters,
  int lMax,
  bool crossover
) {

    // The power spectrum is multiplied by an l-dependent prefactor that comes
    // from the normalization of the Wigner D matrices. This prefactor is
    // mentioned in the arrata of the original SOAP paper: On representing
    // chemical environments, Phys. Rev. B 87, 184115 (2013). Here the square
    // root of the prefactor in the dot-product kernel is used, so that after a
    // possible dot-product the full prefactor is recovered.
    for(int i = 0; i < nCenters; i++){
      int shiftAll = 0;
      for(int j = 0; j < Ts; j++){
       int jdLimit = crossover ? Ts : j+1;
       for(int jd = j; jd < jdLimit; jd++){
        for(int m=0; m <= lMax; m++){
        double prel;
        if(m > 1){prel = PI*sqrt(8.0/(2.0*m+1.0))*PI3;}
        else{prel = PI*sqrt(8.0/(2.0*m+1.0));}
         if(j==jd){
          for(int k = 0; k < Ns; k++){
            for(int kd = k; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = m*m; buffShift < (m+1)*(m+1); buffShift++){
                buffDouble += Cnnd_u(i,j,k,buffShift) * Cnnd_u(i,jd,kd,buffShift);
  	       }
              descriptor_mu(i, shiftAll) = prel*buffDouble;
              shiftAll++;
            }
          }
       } else { 
          for(int k = 0; k < Ns; k++){
            for(int kd = 0; kd < Ns; kd++){
              double buffDouble = 0;
              for(int buffShift = m*m; buffShift < (m+1)*(m+1); buffShift++){
                buffDouble += Cnnd_u(i,j,k,buffShift) * Cnnd_u(i,jd,kd,buffShift);
  	       }
              descriptor_mu(i, shiftAll) = prel*buffDouble;
              shiftAll++;
            }
          }
        }
       } //end ifelse
      }
    }
  }
}
//===========================================================================================
/**
 * Used to calculate the partial power spectrum derivatives.
 */
  void getPDev(
    py::detail::unchecked_mutable_reference<double, 4> &derivatives_mu,
    py::detail::unchecked_reference<double, 2> &positions_u,
    py::detail::unchecked_reference<int, 1> &indices_u,
    CellList &cell_list,
    py::detail::unchecked_reference<double, 5> &CdevX_u,
    py::detail::unchecked_reference<double, 5> &CdevY_u,
    py::detail::unchecked_reference<double, 5> &CdevZ_u,
    py::detail::unchecked_reference<double, 4> &Cnnd_u,
    int Ns,
    int Ts,
    int nCenters,
    int lMax,
    bool crossover
  ) {

  // Loop over all given atomic indices for which the derivatives should be
  // calculated for.
  for (int i_idx = 0; i_idx < indices_u.size(); ++i_idx) {
    int i_atom = indices_u(i_idx);

    // Get all neighbouring centers for the current atom
    double ix = positions_u(i_atom, 0);
    double iy = positions_u(i_atom, 1);
    double iz = positions_u(i_atom, 2);
    CellListResult result = cell_list.getNeighboursForPosition(ix, iy, iz);
    vector<int> indices = result.indices;

    // Loop through all neighbouring centers
    for (size_t j_idx = 0; j_idx < indices.size(); ++j_idx) {
        int i_center = indices[j_idx];
        int shiftAll = 0;
        for(int j = 0; j < Ts; j++) {
            int jdLimit = crossover ? Ts : j+1;
            for(int jd = j; jd < jdLimit; jd++) {
                for(int m=0; m <= lMax; m++) {
                    double prel = m > 1 ? PI*sqrt(8.0/(2.0*m+1.0))*PI3 : PI*sqrt(8.0/(2.0*m+1.0));
                    if (j == jd) {
                        for(int k = 0; k < Ns; k++){
                            for(int kd = k; kd < Ns; kd++){
                            for(int buffShift = m*m; buffShift < (m +1)*(m +1); buffShift++){
                              if( abs(Cnnd_u(i_center,j,k,buffShift)) > 1e-8 ||  abs(Cnnd_u(i_center,j,kd,buffShift)) > 1e-8 ){
                                derivatives_mu(i_center, i_idx, 0, shiftAll) += prel*(Cnnd_u(i_center,j,k,buffShift)*CdevX_u(i_atom, i_center, jd, kd, buffShift)
                                                                                     +Cnnd_u(i_center,j,kd,buffShift)*CdevX_u(i_atom, i_center, jd, k, buffShift));
                                derivatives_mu(i_center, i_idx, 1, shiftAll) += prel*(Cnnd_u(i_center,j,k,buffShift)*CdevY_u(i_atom, i_center, jd, kd, buffShift)
                                                                                     +Cnnd_u(i_center,j,kd,buffShift)*CdevY_u(i_atom, i_center, jd, k, buffShift));
                                derivatives_mu(i_center, i_idx, 2, shiftAll) += prel*(Cnnd_u(i_center,j,k,buffShift)*CdevZ_u(i_atom, i_center, jd, kd, buffShift)
                                                                                     +Cnnd_u(i_center,j,kd,buffShift)*CdevZ_u(i_atom, i_center, jd, k, buffShift));
                                
                            }}
                            shiftAll++;
                            }
                        }
                    } else {
                        for(int k = 0; k < Ns; k++){
                            for(int kd = 0; kd < Ns; kd++){
                            for(int buffShift = m*m; buffShift < (m +1)*(m +1); buffShift++){
                              if( abs(Cnnd_u(i_center,j,k,buffShift)) > 1e-8 ||  abs(Cnnd_u(i_center,jd,kd,buffShift)) > 1e-8) {
                                  derivatives_mu(i_center, i_idx, 0, shiftAll) += prel*(Cnnd_u(i_center,j,k,buffShift)*CdevX_u(i_atom, i_center, jd, kd, buffShift)
                                                                                     +Cnnd_u(i_center,jd,kd,buffShift)*CdevX_u(i_atom, i_center, j, k, buffShift));
                                  derivatives_mu(i_center, i_idx, 1, shiftAll) += prel*(Cnnd_u(i_center,j,k,buffShift)*CdevY_u(i_atom, i_center, jd, kd, buffShift)
                                                                                     +Cnnd_u(i_center,jd,kd,buffShift)*CdevY_u(i_atom, i_center, j, k, buffShift));
                                  derivatives_mu(i_center, i_idx, 2, shiftAll) += prel*(Cnnd_u(i_center,j,k,buffShift)*CdevZ_u(i_atom, i_center, jd, kd, buffShift)
                                                                                     +Cnnd_u(i_center,jd,kd,buffShift)*CdevZ_u(i_atom, i_center, j, k, buffShift));
                            }}
                            shiftAll++;
                            }
                        }
                    }
                }
            }
        }
    }
  }
}
//=================================================================================================================================================================
void soapGTO(
    py::array_t<double> derivatives,
    py::array_t<double> descriptor,
    py::array_t<double> cdevX,
    py::array_t<double> cdevY,
    py::array_t<double> cdevZ,
    py::array_t<double> positions,
    py::array_t<double> centers,
    py::array_t<int> center_indices,
    py::array_t<double> alphasArr,
    py::array_t<double> betasArr,
    py::array_t<int> atomicNumbersArr,
    py::array_t<int> orderedSpeciesArr,
    const double rCut,
    const double cutoffPadding,
    const int nMax,
    const int lMax,
    const double eta,
    py::dict weighting,
    const bool crossover,
    string average,
    py::array_t<int> indices,
    const bool attach,
    const bool return_descriptor,
    const bool return_derivatives,
    CellList cell_list_atoms
) {
  const int totalAN = atomicNumbersArr.shape(0);
  const int nCenters = centers.shape(0);
  auto derivatives_mu = derivatives.mutable_unchecked<4>();
  auto atomicNumbers = atomicNumbersArr.unchecked<1>();
  auto species = orderedSpeciesArr.unchecked<1>();
  int nSpecies = orderedSpeciesArr.shape(0);
  auto indices_u = indices.unchecked<1>();
  double *alphas = (double*)alphasArr.request().ptr;
  double *betas = (double*)betasArr.request().ptr;
  double oOeta = 1.0/eta;
  double oOeta3O2 = sqrt(oOeta*oOeta*oOeta);
  double nMax2 = nMax*nMax;
  auto centers_u = centers.unchecked<2>(); 
  auto center_indices_u = center_indices.unchecked<1>(); 
  auto positions_u = positions.unchecked<2>(); 
  const int nFeatures = crossover
        ? (nSpecies*nMax)*(nSpecies*nMax+1)/2*(lMax+1) 
        : nSpecies*(lMax+1)*((nMax+1)*nMax)/2;
  double* weights = (double*) malloc(sizeof(double)*totalAN);
  double* dx  = (double*)malloc(sizeof(double)*totalAN); double* dy  = (double*)malloc(sizeof(double)*totalAN); double* dz  = (double*)malloc(sizeof(double)*totalAN);
  double* x2  = (double*)malloc(sizeof(double)*totalAN); double* x4  = (double*)malloc(sizeof(double)*totalAN); double* x6  = (double*)malloc(sizeof(double)*totalAN);
  double* x8  = (double*)malloc(sizeof(double)*totalAN); double* x10 = (double*)malloc(sizeof(double)*totalAN); double* x12 = (double*)malloc(sizeof(double)*totalAN);
  double* x14 = (double*)malloc(sizeof(double)*totalAN); double* x16 = (double*)malloc(sizeof(double)*totalAN); double* x18 = (double*)malloc(sizeof(double)*totalAN);
  double* y2  = (double*)malloc(sizeof(double)*totalAN); double* y4  = (double*)malloc(sizeof(double)*totalAN); double* y6  = (double*)malloc(sizeof(double)*totalAN);
  double* y8  = (double*)malloc(sizeof(double)*totalAN); double* y10 = (double*)malloc(sizeof(double)*totalAN); double* y12 = (double*)malloc(sizeof(double)*totalAN);
  double* y14 = (double*)malloc(sizeof(double)*totalAN); double* y16 = (double*)malloc(sizeof(double)*totalAN); double* y18 = (double*)malloc(sizeof(double)*totalAN);
  double* z2  = (double*)malloc(sizeof(double)*totalAN); double* z4  = (double*)malloc(sizeof(double)*totalAN); double* z6  = (double*)malloc(sizeof(double)*totalAN);
  double* z8  = (double*)malloc(sizeof(double)*totalAN); double* z10 = (double*)malloc(sizeof(double)*totalAN); double* z12 = (double*)malloc(sizeof(double)*totalAN);
  double* z14 = (double*)malloc(sizeof(double)*totalAN); double* z16 = (double*)malloc(sizeof(double)*totalAN); double* z18 = (double*)malloc(sizeof(double)*totalAN);
  double* r1  = (double*)malloc(sizeof(double)*totalAN);
  double* r2  = (double*)malloc(sizeof(double)*totalAN); double* r4  = (double*)malloc(sizeof(double)*totalAN); double* r6  = (double*)malloc(sizeof(double)*totalAN);
  double* r8  = (double*)malloc(sizeof(double)*totalAN); double* r10 = (double*)malloc(sizeof(double)*totalAN); double* r12 = (double*)malloc(sizeof(double)*totalAN);
  double* r14 = (double*)malloc(sizeof(double)*totalAN); double* r16 = (double*)malloc(sizeof(double)*totalAN); double* r18 = (double*)malloc(sizeof(double)*totalAN);

  double* r20 = (double*)malloc(sizeof(double)*totalAN); double* x20 = (double*)malloc(sizeof(double)*totalAN); double* y20 = (double*)malloc(sizeof(double)*totalAN);
  double* z20 = (double*)malloc(sizeof(double)*totalAN);
  double* exes = (double*) malloc(sizeof(double)*totalAN);
  // -4 -> no need for l=0, l=1.
  double* preCoef = (double*) malloc(((lMax+1)*(lMax+1)-4)*sizeof(double)*totalAN);
  double* prCofDX;
  double* prCofDY;
  double* prCofDZ;
  if (return_derivatives) {
    prCofDX = (double*) malloc(((lMax+1)*(lMax+1)-4)*sizeof(double)*totalAN);
    prCofDY = (double*) malloc(((lMax+1)*(lMax+1)-4)*sizeof(double)*totalAN);
    prCofDZ = (double*) malloc(((lMax+1)*(lMax+1)-4)*sizeof(double)*totalAN);
  }

  double* bOa = (double*) malloc((lMax+1)*nMax2*sizeof(double));
  double* aOa = (double*) malloc((lMax+1)*nMax*sizeof(double));
  
  // Initialize temporary numpy array for storing the coefficients and the
  // averaged coefficients if inner averaging was requested.
  const int n_coeffs = nSpecies*nMax*(lMax + 1) * (lMax + 1);
  double* cnnd_raw = new double[nCenters*n_coeffs]();
  py::array_t<double> cnnd({nCenters, nSpecies, nMax, (lMax + 1) * (lMax + 1)}, cnnd_raw);
  double* cnnd_ave_raw;
  py::array_t<double> cnnd_ave;
  if (average == "inner") {
      cnnd_ave_raw = new double[n_coeffs]();
      cnnd_ave = py::array_t<double>({1, nSpecies, nMax, (lMax + 1) * (lMax + 1)}, cnnd_ave_raw);
  }

  auto cnnd_u = cnnd.unchecked<4>();
  auto cdevX_u = cdevX.unchecked<5>();
  auto cdevY_u = cdevY.unchecked<5>();
  auto cdevZ_u = cdevZ.unchecked<5>();

  auto cnnd_mu = cnnd.mutable_unchecked<4>(); 
  auto cdevX_mu = cdevX.mutable_unchecked<5>();
  auto cdevY_mu = cdevY.mutable_unchecked<5>();
  auto cdevZ_mu = cdevZ.mutable_unchecked<5>();

  // Initialize binning for atoms and centers
  CellList cell_list_centers(centers, rCut+cutoffPadding);

  // Create a mapping between an atomic index and its internal index in the
  // output. The list of species is already ordered.
  map<int, int> ZIndexMap;
  for (int i = 0; i < species.size(); ++i) {
      ZIndexMap[species(i)] = i;
  }

  getAlphaBetaD(aOa,bOa,alphas,betas,nMax,lMax,oOeta, oOeta3O2);

  // Loop through the centers
  for (int i = 0; i < nCenters; i++) {
    // If computing derivatives with attach=True, index of the center atom is needed
    int centerAtomI = (return_derivatives && attach) ? center_indices_u(i) : -1;

    // Get all neighbouring atoms for the center i
    double ix = centers_u(i, 0); double iy = centers_u(i, 1); double iz = centers_u(i, 2);
    CellListResult result = cell_list_atoms.getNeighboursForPosition(ix, iy, iz);

    // Sort the neighbours by type
    map<int, vector<int>> atomicTypeMap;
    for (const int &idx : result.indices) {int Z = atomicNumbers(idx); atomicTypeMap[Z].push_back(idx);};

    // Loop through neighbours sorted by type
    for (const auto &ZIndexPair : atomicTypeMap) {

      // j is the internal index for this atomic number
      int j = ZIndexMap[ZIndexPair.first];
      int n_neighbours = ZIndexPair.second.size();

      // Save the neighbour distances into the arrays dx, dy and dz
      getDeltaD(dx, dy, dz, positions, ix, iy, iz, ZIndexPair.second);
      getRsZsD(dx, x2, x4, x6, x8, x10, x12, x14, x16, x18, dy, y2, y4, y6, y8, y10, y12, y14, y16, y18, dz, r2, r4, r6, r8, r10, r12, r14, r16, r18,  z2, z4, z6, z8, z10, z12, z14, z16, z18, r20, x20, y20, z20, n_neighbours, lMax);
      getWeights(n_neighbours, r1, r2, true, weighting, weights);
      getCfactorsD(preCoef, prCofDX, prCofDY, prCofDZ, n_neighbours, dx,x2, x4, x6, x8,x10,x12,x14,x16,x18, dy,y2, y4, y6, y8,y10,y12,y14,y16,y18, dz, z2, z4, z6, z8,z10,z12,z14,z16,z18, r2, r4, r6, r8,r10,r12,r14,r16,r18,r20, x20,y20,z20, totalAN, lMax, return_derivatives);
      getCD(cdevX_mu, cdevY_mu, cdevZ_mu, prCofDX, prCofDY, prCofDZ, cnnd_mu, preCoef, dx, dy, dz, r2, weights, bOa, aOa, exes, totalAN, n_neighbours, nMax, nSpecies, lMax, i, centerAtomI, j, ZIndexPair.second, attach, return_derivatives);
    }
  }
  free(dx); free(x2); free(x4); free(x6); free(x8); free(x10); free(x12); free(x14); free(x16); free(x18);
  free(dy); free(y2); free(y4); free(y6); free(y8); free(y10); free(y12); free(y14); free(y16); free(y18);
  free(dz); free(z2); free(z4); free(z6); free(z8); free(z10); free(z12); free(z14); free(z16); free(z18);
  free(r1);
  free(r2); free(r4); free(r6); free(r8); free(r10); free(r12); free(r14); free(r16); free(r18);
  free(r20);
  free(x20);
  free(y20);
  free(z20);
  free(exes); free(preCoef); free(bOa); free(aOa); free(cnnd_raw); free(weights);

  if (return_derivatives) {
    free(prCofDX); free(prCofDY); free(prCofDZ);
  }

  // Calculate the descriptor value if requested
  if (return_descriptor) {
    auto descriptor_mu = descriptor.mutable_unchecked<2>();

    // If inner averaging is requested, average the coefficients over the
    // centers (axis 0) before calculating the power spectrum.
    if (average == "inner") {
        auto cnnd_ave_mu = cnnd_ave.mutable_unchecked<4>(); 
        auto cnnd_ave_u = cnnd_ave.unchecked<4>(); 
        for (int i = 0; i < nCenters; i++) {
            for (int j = 0; j < nSpecies; j++) {
                for (int k = 0; k < nMax; k++) {
                    for (int l = 0; l < (lMax + 1) * (lMax + 1); l++) {
                        cnnd_ave_mu(0, j, k, l) += cnnd_u(i, j, k, l);
                    }
                }
            }
        }
        for (int j = 0; j < nSpecies; j++) {
            for (int k = 0; k < nMax; k++) {
                for (int l = 0; l < (lMax + 1) * (lMax + 1); l++) {
                    cnnd_ave_mu(0, j, k, l) = cnnd_ave_mu(0, j, k, l) / (double)nCenters;
                }
            }
        }
        getPD(descriptor_mu, cnnd_ave_u, nMax, nSpecies, 1, lMax, crossover);
        delete [] cnnd_ave_raw;
    // If outer averaging is requested, average the power spectrum across the
    // centers.
    } else if (average == "outer") {
        // We allocate the memory and give array_t a pointer to it. This way
        // the memory is owned and freed by C++.
        double* ps_temp_raw = new double[nCenters*nFeatures]();
        py::array_t<double> ps_temp({nCenters, nFeatures}, ps_temp_raw);
        auto ps_temp_mu = ps_temp.mutable_unchecked<2>();
        getPD(ps_temp_mu, cnnd_u, nMax, nSpecies, nCenters, lMax, crossover);
        for (int i = 0; i < nCenters; i++) {
            for (int j = 0; j < nFeatures; j++) {
                descriptor_mu(0, j) += ps_temp_mu(i, j);
            }
        }
        for (int j = 0; j < nFeatures; j++) {
            descriptor_mu(0, j) = descriptor_mu(0, j) / (double)nCenters;
        }
        delete [] ps_temp_raw;
    // Regular power spectrum without averaging
    } else {
        getPD(descriptor_mu, cnnd_u, nMax, nSpecies, nCenters, lMax, crossover);
    }
  }

  // Calculate the derivatives
  if (return_derivatives) {
    getPDev(derivatives_mu, positions_u, indices_u, cell_list_centers, cdevX_u, cdevY_u, cdevZ_u, cnnd_u, nMax, nSpecies, nCenters, lMax, crossover);
  }

  return;
}
