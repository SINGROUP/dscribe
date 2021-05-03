#ifndef WEIGHTING_H
#define WEIGHTING_H

#include <cmath>
#include <pybind11/pybind11.h>

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
 * Used to calculate the Gaussian weights for each neighbouring atom. Provide
 * either r1s (=r) or r2s (=r^2) and use the boolean "squared" to indicate if
 * r1s should be calculated from r2s.
 */
void getWeights(int size, double* r1s, double* r2s, const bool squared, const pybind11::dict &weighting, double* weights);

#endif
