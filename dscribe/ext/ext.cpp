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
#include "soapGTO.h"
#include "soapGeneral.h"
#include "acsf.h"
#include "mbtr.h"

namespace py = pybind11;
using namespace std;

// Notice that the name of the first argument to the module macro needs to
// correspond to the file name!
PYBIND11_MODULE(ext, m) {
    // SOAP
    m.def("soap_gto", &soapGTO, "SOAP with gaussian type orbital radial basis set.");
    m.def("soap_general", &soapGeneral, "SOAP with a general radial basis set.");

    // ACSF
    py::class_<ACSF>(m, "ACSFWrapper")
        .def(py::init<float , vector<vector<float> > , vector<float> , vector<vector<float> > , vector<vector<float> > , vector<int> >())
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


        .def_property("rcut", &ACSF::getRCut, &ACSF::setRCut)
        .def_property("g3_params", &ACSF::getG3Params, &ACSF::setG3Params)
        .def_property("g4_params", &ACSF::getG4Params, &ACSF::setG4Params)
        .def_property("g5_params", &ACSF::getG5Params, &ACSF::setG5Params)
        .def_property("atomic_numbers", &ACSF::getAtomicNumbers, &ACSF::setAtomicNumbers)

       .def(py::pickle(
        [](const ACSF &p) { // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.rCut, p.g2Params, p.g3Params, p.g4Params, p.g5Params, p.atomicNumbers);
        },
        [](py::tuple t) { // __setstate__
            if (t.size() != 6)
                throw std::runtime_error("Invalid state!");

            /* Create a new C++ instance */
            ACSF p(t[0].cast<float>(), t[1].cast<vector<vector<float> >>(), t[2].cast<vector<float>>(), t[3].cast<vector<vector<float> >>(), t[4].cast<vector<vector<float> >>(), t[5].cast<vector<int>>());
            /* Assign any additional state */
            //p.setExtra(t[1].cast<int>());

            return p;
        }
        ));
 
    //MBTR
    py::class_<MBTR>(m, "MBTRWrapper")
        .def(py::init< map<int,int>, int , vector<vector<int>>  >())
        //.def(py::init< map<int,int> atomicNumberToIndexMap, int interactionLimit, vector<vector<int>> cellIndices >())
        .def("get_k1", &MBTR::getK1)
        .def("get_k2", &MBTR::getK2)
        .def("get_k3", &MBTR::getK3)
        .def("get_k2_local", &MBTR::getK2Local)
        .def("get_k3_local", &MBTR::getK3Local)

        ;


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
}
