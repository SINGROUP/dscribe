#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // Enables easy access to numpy arrays
#include <pybind11/stl.h>    // Enables automatic type conversion from C++ containers to python
#include "celllist.h"

namespace py = pybind11;
using namespace std;

// Notice that the name of the first argument to the module macro needs to
// correspond to the file name!
PYBIND11_MODULE(libutils, m) {
    py::class_<CellList>(m, "CellList")
        .def(py::init<py::array_t<double>, double>())
        .def("get_neighbours_for_index", &CellList::getNeighboursForIndex)
        .def("get_neighbours_for_position", &CellList::getNeighboursForPosition);
}
