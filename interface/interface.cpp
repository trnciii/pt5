#include <pybind11/pybind11.h>
#include "pt5.hpp"


PYBIND11_MODULE(core, m) {
    m
    .def("add", &pt5::add)
    .def("nothing", &pt5::nothing);
}