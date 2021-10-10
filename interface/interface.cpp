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

	const uint32_t nVerts = pV.shape(0);

	auto pIndex = _pIndex.mutable_unchecked<2>();
	auto pmID = _pmID.mutable_unchecked<1>();
	assert(pIndex.shape(1) == 3);
	assert(pIndex.shape(0) == pmID.shape(0));

	const uint32_t nFaces = pIndex.shape(0);

	auto pmSlot = _pmSlot.mutable_unchecked<1>();



	std::vector<float3> cV(nVerts);
	memcpy(cV.data(), (float3*)pV.data(0,0), nVerts*sizeof(float3));


	std::vector<float3> cN(nVerts);
	memcpy(cN.data(), (float3*)pN.data(0,0), nVerts*sizeof(float3));


	std::vector<uint3> cIndex(nFaces);
	memcpy(cIndex.data(), (uint3*)pIndex.data(0,0), nFaces*sizeof(uint3));


	std::vector<uint32_t> cMID(nFaces);
	memcpy(cMID.data(), pmID.data(0), nFaces*sizeof(uint32_t));


	std::vector<uint32_t> cMSlot(pmSlot.shape(0));
	memcpy(cMSlot.data(), pmSlot.data(0), pmSlot.shape(0)*sizeof(uint32_t));

	return pt5::TriangleMesh{cV, cN, cIndex, cMID, cMSlot};
}



void cuda_sync(){
	CUDA_SYNC_CHECK();
}


PYBIND11_MODULE(core, m) {
	using namespace pt5;

	py::class_<View>(m, "View")
		.def(py::init<int, int>())
		.def("downloadImage", &View::downloadImage)
		.def("showWindow", &View::showWindow)
		.def_readwrite("width", &View::width)
		.def_readwrite("height", &View::height)
		.def_readwrite("pixels", &View::pixels)
		.def_property_readonly("pixels",
			[](View& self){
			return (py::array_t<float>)py::buffer_info(
				self.pixels.data(),
				sizeof(float),
				py::format_descriptor<float>::format(),
				3,
				{self.height, self.width, 4},
				{self.width*4*sizeof(float), 4*sizeof(float), sizeof(float)}
				);
			});


	py::class_<PathTracerState>(m, "PathTracer")
		.def(py::init<>())
		.def("init", &PathTracerState::init)
		.def("setScene", &PathTracerState::setScene)
		.def("initLaunchParams", &PathTracerState::initLaunchParams)
		.def("render", &PathTracerState::render);


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
		.def_property("albedo", PROPERTY_FLOAT3(Material, albedo))
		.def_property("emission", PROPERTY_FLOAT3(Material, emission));


	py::class_<TriangleMesh>(m, "TriangleMesh");
	m.def("createTriangleMesh", &createTriangleMesh);

	m.def("cuda_sync", &cuda_sync);

}