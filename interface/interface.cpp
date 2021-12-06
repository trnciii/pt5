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
		auto r = x.unchecked<1>();                       \
		self.member = make_float3(r(0), r(1), r(2));     \
	}


template <typename T>
std::vector<T> toSTDVector(const py::array_t<T>& x){
	if(x.size()==0) return std::vector<T>(0);
	else return std::vector<T>(x.data(0), x.data(0)+x.size());
}

const std::vector<py::tuple> float3_dtype{
	py::make_tuple("x", "<f4"),
	py::make_tuple("y", "<f4"),
	py::make_tuple("z", "<f4")
};

const std::vector<py::tuple> uint3_dtype{
	py::make_tuple("x", "<i4"),
	py::make_tuple("y", "<i4"),
	py::make_tuple("z", "<i4")
};



PYBIND11_MODULE(core, m) {
	using namespace pt5;

	py::class_<View>(m, "View")
		.def(py::init<int, int>())
		.def("downloadImage", &View::downloadImage)
		.def("createGLTexture", &View::createGLTexture)
		.def("destroyGLTexture", &View::destroyGLTexture)
		.def("updateGLTexture", &View::updateGLTexture)
		.def("clear", [](View& self, py::array_t<float> c){
			auto r = c.unchecked<1>();
			assert(r.shape(0) == 4);
			self.clear(make_float4(r(0), r(1), r(2), r(3)));
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
		.def_property_readonly("running",[](PathTracerState& self){return self.running();});


	py::class_<Scene>(m, "Scene")
		.def(py::init<>())
		.def_readwrite("materials", &Scene::materials)
		.def_readwrite("meshes", &Scene::meshes)
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
				assert(x.ndim()==2 && x.shape(0) == 3 && x.shape(1) == 3);
				memcpy(&self.toWorld, x.data(0,0), 9*sizeof(float));
			})
		.def_readwrite("focalLength", &Camera::focalLength);


	py::class_<Material>(m, "Material")
		.def(py::init<>())
		.def_property("albedo", PROPERTY_FLOAT3(Material, albedo))
		.def_property("emission", PROPERTY_FLOAT3(Material, emission));


	PYBIND11_NUMPY_DTYPE(float2, x, y);
	PYBIND11_NUMPY_DTYPE(float3, x, y, z);
	PYBIND11_NUMPY_DTYPE(uint3, x, y, z);
	PYBIND11_NUMPY_DTYPE(Vertex, p, n);
	PYBIND11_NUMPY_DTYPE(Face, vertices, uv, smooth, material);

	m.attr("Vertex_dtype") = std::vector<py::tuple>{
		py::make_tuple("p", float3_dtype),
		py::make_tuple("n", float3_dtype)
	};

	m.attr("Face_dtype") = std::vector<py::tuple>{
		py::make_tuple("vertices", uint3_dtype),
		py::make_tuple("uv", uint3_dtype),
		py::make_tuple("smooth", "i1"),
		py::make_tuple("material", "<i4")
	};

	py::class_<TriangleMesh>(m, "TriangleMesh")
		.def(py::init([](
			const py::array_t<Vertex>& v,
			const py::array_t<Face>& f,
			const py::array_t<float>& uv,
			const py::array_t<uint32_t>& m)
		{
			return TriangleMesh(
				toSTDVector(v),
				toSTDVector(f),
				std::vector<float2>((float2*)uv.data(0,0), (float2*)uv.data(0,0)+uv.shape(0)),
				toSTDVector(m));
		}));


	m.def("cuda_sync", [](){CUDA_SYNC_CHECK();});

}