#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pt5.hpp"


namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    using namespace pt5;

    py::class_<PathTracerState>(m, "PathTracer")
        .def(py::init<>())
        .def("init", &PathTracerState::init)
        .def("setScene", &PathTracerState::setScene)
        .def("initLaunchParams", &PathTracerState::initLaunchParams)
        .def("render", &PathTracerState::render)
        .def("pixels", &PathTracerState::pixels);

}