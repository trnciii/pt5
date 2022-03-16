#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cassert>

#include "pt5.hpp"
#include "./util.hpp"

namespace py = pybind11;


void init_material(py::module_ &);
void init_scene(py::module_ &);


PYBIND11_MODULE(core, m) {
	using namespace pt5;

	init_material(m);
	init_scene(m);

	py::class_<View>(m, "View")
		.def(py::init<int, int>())
		.def("downloadImage", &View::downloadImage)
		.def("createGLTexture", &View::createGLTexture)
		.def("destroyGLTexture", &View::destroyGLTexture)
		.def("updateGLTexture", &View::updateGLTexture)
		.def("clear", [](View& self, py::array_t<float> c){
			self.clear(make_float4(c.at(0), c.at(1), c.at(2), c.at(3)));
		})
		.def_property_readonly("GLTexture", &View::GLTexture)
		.def_property_readonly("hasGLTexture", &View::hasGLTexture)
		.def_property_readonly("size",
			[](View& self){
				return py::make_tuple(self.size().x, self.size().y);
			})
		.def_property_readonly("pixels",
			[](View& self){
				return (py::array_t<float>)py::buffer_info(
					self.pixels.data(),
					sizeof(float),
					py::format_descriptor<float>::format(),
					3,
					{(int)self.size().y, (int)self.size().x, 4},
					{(int)self.size().x*4*sizeof(float), 4*sizeof(float), sizeof(float)}
					);
			});


	py::class_<PathTracerState>(m, "PathTracer")
		.def(py::init<>())
		.def("setScene", &PathTracerState::setScene)
		.def("removeScene", &PathTracerState::removeScene)
		.def("render", &PathTracerState::render)
		.def("sync", &PathTracerState::sync)
		.def_property_readonly("running",[](PathTracerState& self){return self.running();});


	m.def("cuda_sync", [](){CUDA_SYNC_CHECK();});
	m.def("linear_to_sRGB", py::vectorize(linear_to_sRGB));

}