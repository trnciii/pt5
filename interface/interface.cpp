#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cassert>

#include "pt5.hpp"


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
		auto r = x.mutable_unchecked<1>();               \
		assert(r.shape(0) == 3);                         \
		memcpy(&self.member, x.data(0), sizeof(float3)); \
	}


pt5::TriangleMesh createTriangleMesh(
	py::array_t<float>& _pV,
	py::array_t<float>& _pN,
	py::array_t<uint>& _pIndex,
	py::array_t<uint32_t>& _pmID,
	py::array_t<uint32_t>& _pmSlot)
{
	auto pV = _pV.mutable_unchecked<2>();
	auto pN = _pN.mutable_unchecked<2>();
	assert(pV.shape(1) == 3);
	assert(pN.shape(1) == 3);
	assert(pV.shape(0) == pN.shape(0));

	auto pIndex = _pIndex.mutable_unchecked<2>();
	auto pmID = _pmID.mutable_unchecked<1>();
	assert(pIndex.shape(1) == 3);
	assert(pIndex.shape(0) == pmID.shape(0));

	auto pmSlot = _pmSlot.mutable_unchecked<1>();


	std::vector<pt5::Vertex> cv(pV.shape(0));
	for(uint32_t i=0; i<pV.shape(0); i++){
		cv[i] = pt5::Vertex{*(float3*)pV.data(i,0), *(float3*)pN.data(i,0)};
	}


	std::vector<pt5::Face> cf(pIndex.shape(0));
	for(uint32_t i=0; i<pIndex.shape(0); i++){
		cf[i] = pt5::Face{*(uint3*)pIndex.data(i,0), *pmID.data(i)};
	}


	std::vector<uint32_t> cm(pmSlot.shape(0));
	memcpy(cm.data(), pmSlot.data(0), pmSlot.shape(0)*sizeof(uint32_t));

	return pt5::TriangleMesh{cv, cf, cm};
}


PYBIND11_MODULE(core, m) {
	using namespace pt5;

	py::class_<PathTracerState>(m, "PathTracer")
		.def(py::init<>())
		.def("init", &PathTracerState::init)
		.def("setScene", &PathTracerState::setScene)
		.def("initLaunchParams", &PathTracerState::initLaunchParams)
		.def("render", &PathTracerState::render)
		.def("pixels", &PathTracerState::pixels);


	py::class_<Scene>(m, "Scene")
		.def(py::init<>())
		.def_readwrite("materials", &Scene::materials)
		.def_readwrite("meshes", &Scene::meshes)
		.def_readwrite("camera", &Scene::camera)
		.def_property("background", PROPERTY_FLOAT3(Scene, background));


	py::class_<Camera>(m, "Camera")
		.def(py::init<>())
		.def_property("position", PROPERTY_FLOAT3(Camera, position))
		.def_property("toWorld",
			[](const Camera& self){
				return (py::array_t<float>)py::buffer_info(
					(float*)&self.toWorld,
					sizeof(float),
					py::format_descriptor<float>::format(),
					2,
					{3, 3},
					{sizeof(float3), sizeof(float)}
					);
			},
			[](Camera& self, py::array_t<float>& x){
				auto r = x.mutable_unchecked<2>();
				assert(x.shape(0) == 3 && x.shape(1) == 3);
				memcpy(&self.toWorld, r.data(0,0), 9*sizeof(float));
			})
		.def_readwrite("focalLength", &Camera::focalLength);


	py::class_<Material>(m, "Material")
		.def(py::init<>())
		.def_property("color", PROPERTY_FLOAT3(Material, color));


	py::class_<TriangleMesh>(m, "TriangleMesh");
	m.def("createTriangleMesh", &createTriangleMesh);

}