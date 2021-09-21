#include <pybind11/pybind11.h>
#include "pt5.hpp"


PYBIND11_MODULE(pypt5, m) {
    m
    .def("add", &add)
    .def("nothing", &nothing);
}