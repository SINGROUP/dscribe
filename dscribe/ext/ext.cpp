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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // Enables easy access to numpy arrays
#include <pybind11/stl.h>    // Enables automatic type conversion from C++ containers to python
#include "celllist.h"
#include "cm.h"
#include "soap.h"
#include "acsf.h"
#include "mbtr.h"
#include "geometry.h"

namespace py = pybind11;
using namespace std;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

// Notice that the name of the first argument to the module macro needs to
// correspond to the file name!
PYBIND11_MODULE(ext, m) {
    // CoulombMatrix
    py::class_<CoulombMatrix>(m, "CoulombMatrix")
        .def(py::init<unsigned int, string, double, int>())
        .def("create", &CoulombMatrix::create)
        .def("derivatives_numerical", &CoulombMatrix::derivatives_numerical)
        .def(py::pickle(
            [](const CoulombMatrix &p) {
                return py::make_tuple(p.n_atoms_max, p.permutation, p.sigma, p.seed);
            },
            [](py::tuple t) {
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");
                CoulombMatrix p(
                    t[0].cast<unsigned int>(),
                    t[1].cast<string>(),
                    t[2].cast<double>(),
                    t[3].cast<int>()
                );
                return p;
            }
        ));

    // SOAP
    py::class_<SOAPGTO>(m, "SOAPGTO")
        .def(py::init<double, int, int, double, py::dict, bool, string, double, py::array_t<double>, py::array_t<double>, py::array_t<int>, bool>())
        .def("create", overload_cast_<py::array_t<double>, py::array_t<double>, py::array_t<int>, py::array_t<double> >()(&SOAPGTO::create, py::const_))
        .def("create", overload_cast_<py::array_t<double>, py::array_t<double>, py::array_t<int>, py::array_t<double>, py::array_t<bool>, py::array_t<double> >()(&SOAPGTO::create, py::const_))
        .def("create", overload_cast_<py::array_t<double>, py::array_t<double>, py::array_t<int>, py::array_t<double>, CellList>()(&SOAPGTO::create, py::const_))
        .def("derivatives_numerical", &SOAPGTO::derivatives_numerical)
        .def("derivatives_analytical", &SOAPGTO::derivatives_analytical);
    py::class_<SOAPPolynomial>(m, "SOAPPolynomial")
        .def(py::init<double, int, int, double, py::dict, bool, string, double, py::array_t<double>, py::array_t<double>, py::array_t<int>, bool >())
        .def("create", overload_cast_<py::array_t<double>, py::array_t<double>, py::array_t<int>, py::array_t<double> >()(&SOAPPolynomial::create, py::const_))
        .def("create", overload_cast_<py::array_t<double>, py::array_t<double>, py::array_t<int>, py::array_t<double>, py::array_t<bool>, py::array_t<double> >()(&SOAPPolynomial::create, py::const_))
        .def("create", overload_cast_<py::array_t<double>, py::array_t<double>, py::array_t<int>, py::array_t<double>, CellList>()(&SOAPPolynomial::create, py::const_))
        .def("derivatives_numerical", &SOAPPolynomial::derivatives_numerical);

    // ACSF
    py::class_<ACSF>(m, "ACSFWrapper")
        .def(py::init<double , vector<vector<double> > , vector<double> , vector<vector<double> > , vector<vector<double> > , vector<int> >())
        .def(py::init<>())
        .def("create", &ACSF::create)
        .def("set_g2_params", &ACSF::setG2Params)
        .def("get_g2_params", &ACSF::getG2Params)
        .def_readwrite("n_types", &ACSF::nTypes)
        .def_readwrite("n_type_pairs", &ACSF::nTypePairs)
        .def_readwrite("n_g2", &ACSF::nG2)
        .def_readwrite("n_g3", &ACSF::nG3)
        .def_readwrite("n_g4", &ACSF::nG4)
        .def_readwrite("n_g5", &ACSF::nG5)
        .def_property("r_cut", &ACSF::getRCut, &ACSF::setRCut)
        .def_property("g3_params", &ACSF::getG3Params, &ACSF::setG3Params)
        .def_property("g4_params", &ACSF::getG4Params, &ACSF::setG4Params)
        .def_property("g5_params", &ACSF::getG5Params, &ACSF::setG5Params)
        .def_property("atomic_numbers", &ACSF::getAtomicNumbers, &ACSF::setAtomicNumbers)

        .def(py::pickle(
            [](const ACSF &p) {
                return py::make_tuple(p.rCut, p.g2Params, p.g3Params, p.g4Params, p.g5Params, p.atomicNumbers);
            },
            [](py::tuple t) {
                if (t.size() != 6)
                    throw std::runtime_error("Invalid state!");
                ACSF p(
                    t[0].cast<double>(),
                    t[1].cast<vector<vector<double> >>(),
                    t[2].cast<vector<double>>(),
                    t[3].cast<vector<vector<double> >>(),
                    t[4].cast<vector<vector<double> >>(),
                    t[5].cast<vector<int>>()
                );
                return p;
            }
        ));
 
    // MBTR
    py::class_<MBTR>(m, "MBTRWrapper")
        .def(py::init< map<int,int>, int , vector<vector<int>>  >())
        .def("get_k1", &MBTR::getK1)
        .def("get_k2", &MBTR::getK2)
        .def("get_k3", &MBTR::getK3)
        .def("get_k2_local", &MBTR::getK2Local)
        .def("get_k3_local", &MBTR::getK3Local);

    // CellList
    py::class_<CellList>(m, "CellList")
        .def(py::init<py::array_t<double>, double>())
        .def("get_neighbours_for_index", &CellList::getNeighboursForIndex)
        .def("get_neighbours_for_position", &CellList::getNeighboursForPosition);
    py::class_<CellListResult>(m, "CellListResult")
        .def(py::init<>())
        .def_readonly("indices", &CellListResult::indices)
        .def_readonly("distances", &CellListResult::distances)
        .def_readonly("distances_squared", &CellListResult::distancesSquared);

    // Geometry
    m.def("extend_system", &extend_system, "Create a periodically extended system.");
    py::class_<ExtendedSystem>(m, "ExtendedSystem")
        .def(py::init<>())
        .def_readonly("positions", &ExtendedSystem::positions)
        .def_readonly("atomic_numbers", &ExtendedSystem::atomic_numbers)
        .def_readonly("indices", &ExtendedSystem::indices);
}
