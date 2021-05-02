#include <cmath>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
using namespace std;

/**
 * Polynomial weighting of the form:
 * w(r) = c * (1 + 2(r/r0)^3 - 3(r/r0)^2)^m, if r <= r0, 0 otherwise
 */
inline double weightPoly(const double r, const double r0, const double c, const double m)
{
    if (r > r0) {
        return 0;
    }
    const double rr0 = r / r0;
    const double rr02 = rr0 * rr0;
    const double rr03 = rr02 * rr0;
    return c * pow(1 + 2*rr03 - 3*rr02, m);
}
/**
 * Power function weighting of the form:
 * w(r) = c / (d + (r/r0)^m)
 */
inline double weightPow(const double r, const double r0, const double c, const double d, const double m)
{
    const double rr0 = r / r0;
    return c / (d + pow(rr0, m));
}
/**
 * Exponential weighting of the form:
 * w(r) = c / (d + e^(-r/r0))
 */
inline double weightExp(const double r, const double r0, const double c, const double d)
{
    const double rr0 = r / r0;
    return c / (d + exp(-rr0));
}
/**
 * Used to calculate the Gaussian weights for each neighbouring atom.
 */
void getWeights(int size, double* r2, const py::dict &weighting, double* weights) {
    // No weighting specified
    if (!weighting.contains("function") && !weighting.contains("w0")) {
        for (int i = 0; i < size; i++) {
            weights[i] = 1;
        }
    } else {
        // No weighting function, only w0
        if (!weighting.contains("function") && weighting.contains("w0")) {
            double w0 = weighting["w0"].cast<double>();
            for (int i = 0; i < size; i++) {
                double r = sqrt(r2[i]);
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
                    double r = sqrt(r2[i]);
                    if (r == 0) {
                        weights[i] = w0;
                    } else {
                        weights[i] = func(r);
                    }
                }
            // Weighting function only
            } else {
                for (int i = 0; i < size; i++) {
                    double r = sqrt(r2[i]);
                    weights[i] = func(r);
                }
            }
        }
    }
}
