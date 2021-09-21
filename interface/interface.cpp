#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pt5.hpp"
#include "util.hpp"


namespace py = pybind11;

PYBIND11_MODULE(core, m) {
    using namespace pt5;

    py::class_<PathTracerState>(m, "PathTracer")
        .def(py::init<>())
        .def("buildSBT", &PathTracerState::buildSBT)
        .def("initLaunchParams", &PathTracerState::initLaunchParams)
        .def("render", &PathTracerState::render)
        .def("pixels", &PathTracerState::pixels);

    m.def("writeImage", &writeImage);
}