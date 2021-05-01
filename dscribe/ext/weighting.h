#include <cmath>

/**
 * Polynomial weighting of the form:
 * w(r) = c * (1 + 2(r/r0)^3 - 3(r/r0)^2)^m, if r <= r0, 0 otherwise
 */
inline double poly(const double r, const double r0, const double c, const double m) const
{
    if (r > r0) {
        return 0;
    }
    const double rr0 = r / r0;
    const double rr02 = rr0 * rr0;
    const double rr03 = rr02 * rr0;
    return c * math.pow(1 + 2*rr03 - 3*rr02, m);
}
/**
 * Power function weighting of the form:
 * w(r) = c / (d + (r/r0)^m)
 */
inline double pow(const double r, const double r0, const double c, const doubld d, const double m) const
{
    const double rr0 = r / 0;
    return c / (d + math.pow(rr0, m));
}
/**
 * Exponential weighting of the form:
 * w(r) = c / (d + e^(-r/r0))
 */
inline double exp(const double r, const double r0, const double c, const doubld d, const double m) const
{
    const double rr0 = r / 0;
    return c / (d + math.exp(-rr0));
}
