#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cassert>

#include "pt5.hpp"
#include "./util.hpp"

namespace py = pybind11;


void init_scene(py::module_& m) {
	using namespace pt5;

	py::class_<Scene>(m, "Scene")
		.def(py::init<>())
		.def_readwrite("materials", &Scene::materials)
		.def_readwrite("meshes", &Scene::meshes)
		.def_readwrite("images", &Scene::images)
		.def_readwrite("background", &Scene::background);


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
				assert(x.ndim()==2 && x.shape(0) == 3 && x.shape(1) == 3);
				memcpy(&self.toWorld, x.data(), 9*sizeof(float));
			})
		.def_readwrite("focalLength", &Camera::focalLength);



	PYBIND11_NUMPY_DTYPE(float2, x, y);
	PYBIND11_NUMPY_DTYPE(float3, x, y, z);
	PYBIND11_NUMPY_DTYPE(uint3, x, y, z);
	PYBIND11_NUMPY_DTYPE(Vertex, p, n);
	PYBIND11_NUMPY_DTYPE(Face, vertices, uv, smooth, material);


	py::class_<TriangleMesh>(m, "TriangleMesh")
		.def(py::init([](
			const py::array_t<Vertex>& v,
			const py::array_t<Face>& f,
			const py::array_t<float>& uv,
			const py::array_t<uint32_t>& m)
		{
			return (TriangleMesh){
				toSTDVector(v),
				toSTDVector(f),
				std::vector<float2>((float2*)uv.data(), (float2*)uv.data()+uv.shape(0)),
				toSTDVector(m)
			};
		}));


	py::class_<Image>(m, "Image")
		.def(py::init([](const py::array_t<float>& data){
			assert(data.ndim() == 3);
			assert(data.shape(2) == 4);
			return Image{
				{static_cast<uint>(data.shape(1)), static_cast<uint>(data.shape(0))},
				std::vector<float4>(
					(float4*)data.data(),
					(float4*)data.data() + (data.shape(0)*data.shape(1)))
			};
		}));

}