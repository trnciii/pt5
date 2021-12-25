#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


namespace py = pybind11;


#define PROPERTY_FLOAT3(Class, member)               \
	[](const Class& self){                             \
		return (py::array_t<float>)py::buffer_info(      \
			(float*)&self.member,                          \
			sizeof(float),                                 \
			py::format_descriptor<float>::format(),        \
			1,{3},{sizeof(float)}                          \
			);                                             \
	},                                                 \
	[](Class& self, py::array_t<float>& x){            \
		auto r = x.unchecked<1>();                       \
		self.member = make_float3(r(0), r(1), r(2));     \
	}


template <typename T>
std::vector<T> toSTDVector(const py::array_t<T>& x){
	if(x.size()==0) return std::vector<T>(0);
	else return std::vector<T>(x.data(), x.data()+x.size());
}