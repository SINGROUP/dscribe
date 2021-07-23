#include <functional>
#include "weighting.h"
namespace py = pybind11;
using namespace std;

/**
 * Used to calculate the Gaussian weights for each neighbouring atom. Provide
 * either r1s (=r) or r2s (=r^2) and use the boolean "squared" to indicate if
 * r1s should be calculated from r2s.
 */
void getWeights(int size, double* r1s, double* r2s, const bool squared, const py::dict &weighting, double* weights) {
    // No weighting specified
    if (!weighting.contains("function") && !weighting.contains("w0")) {
        for (int i = 0; i < size; i++) {
            weights[i] = 1;
        }
    } else {
        // No weighting function, only w0
        if (squared) {
            for (int i = 0; i < size; i++) {
                r1s[i] = sqrt(r2s[i]);
            }
        }
        if (!weighting.contains("function") && weighting.contains("w0")) {
            double w0 = weighting["w0"].cast<double>();
            for (int i = 0; i < size; i++) {
                double r = r1s[i];
                if (r == 0) {
                    weights[i] = w0;
                } else {
                    weights[i] = 1;
                }
            }
        } else {
            function<double (double)> func;
            string fname = weighting["function"].cast<string>();
            if (fname == "poly") {
                double r0 = weighting["r0"].cast<double>();
                double c = weighting["c"].cast<double>();
                double m = weighting["m"].cast<double>();
                func = [r0, c, m](double r) {return weightPoly(r, r0, c, m);};
            } else if (fname == "pow") {
                double r0 = weighting["r0"].cast<double>();
                double c = weighting["c"].cast<double>();
                double d = weighting["d"].cast<double>();
                double m = weighting["m"].cast<double>();
                func = [r0, c, d, m](double r) {return weightPow(r, r0, c, d, m);};
            } else if (fname == "exp") {
                double r0 = weighting["r0"].cast<double>();
                double c = weighting["c"].cast<double>();
                double d = weighting["d"].cast<double>();
                func = [r0, c, d](double r) {return weightExp(r, r0, c, d);};
            }
            // Weighting function and w0
            if (weighting.contains("w0")) {
                double w0 = weighting["w0"].cast<double>();
                for (int i = 0; i < size; i++) {
                    double r = r1s[i];
                    if (r == 0) {
                        weights[i] = w0;
                    } else {
                        weights[i] = func(r);
                    }
                }
            // Weighting function only
            } else {
                for (int i = 0; i < size; i++) {
                    double r = r1s[i];
                    weights[i] = func(r);
                }
            }
        }
    }
}
