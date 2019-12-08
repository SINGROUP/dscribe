#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // Enables easy access to numpy arrays
#include <pybind11/stl.h>    // Enables automatic type conversion from C++ containers to python
#include "soapGTO.h"         // Enables automatic type conversion from C++ containers to python

namespace py = pybind11;
using namespace std;

// Notice that the name of the first argument to the module macro needs to
// correspond to the file name!
PYBIND11_MODULE(libsoap, m) {
    m.def("soap_gto", &soapGTO, "SOAP with gaussian type orbital radial basis set.");
}
