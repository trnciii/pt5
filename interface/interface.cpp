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
	else return std::vector<T>(x.data(), x.data()+x.size());
}


PYBIND11_MODULE(core, m) {
	using namespace pt5;

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
		.def_property_readonly("running",[](PathTracerState& self){return self.running();});


	py::class_<Scene>(m, "Scene")
		.def(py::init<>())
		.def_property("materials",
			[](const Scene& self){
				py::list li;
				for(const std::shared_ptr<Material>& m: self.materials){
					if(m->type() == MaterialType::Diffuse)
						li.append(*(MTLData_Diffuse*)m->ptr());
					else if(m->type() == MaterialType::Emission)
						li.append(*(MTLData_Emission*)m->ptr());
					else
						li.append("error");
				}
				return li;
			},
			[](Scene& self, const py::list& li){
				std::vector<std::shared_ptr<Material>> mtls;
				for(const py::handle& obj : li){
					if(py::isinstance<MTLData_Diffuse>(obj))
						mtls.push_back(abstract_material(obj.cast<MTLData_Diffuse>()));
					else if(py::isinstance<MTLData_Emission>(obj))
						mtls.push_back(abstract_material(obj.cast<MTLData_Emission>()));
					else std::cout <<"error: " <<obj <<std::endl;
				}
				self.materials = mtls;
			})
		.def_readwrite("meshes", &Scene::meshes)
		.def_readwrite("textures", &Scene::textures)
		.def_readwrite("images", &Scene::images)
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
				memcpy(&self.toWorld, x.data(), 9*sizeof(float));
			})
		.def_readwrite("focalLength", &Camera::focalLength);



	py::class_<MTLData_Diffuse>(m, "MTLData_Diffuse")
		.def(py::init<>())
		.def(py::init([](const py::array_t<float>& c, uint32_t t=0){
			return MTLData_Diffuse{make_float3(c.at(0), c.at(1), c.at(2)), t};
		}))
		.def_property("color", PROPERTY_FLOAT3(MTLData_Diffuse, color))
		.def_readwrite("texture", &MTLData_Diffuse::texture);

	py::class_<MTLData_Emission>(m, "MTLData_Emission")
		.def(py::init<>())
		.def(py::init([](const py::array_t<float>& c, uint32_t t=0){
			return MTLData_Emission{make_float3(c.at(0), c.at(1), c.at(2)), t};
		}))
		.def_property("color", PROPERTY_FLOAT3(MTLData_Emission, color))
		.def_readwrite("texture", &MTLData_Emission::texture);


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


	py::class_<Texture>(m, "Texture")
		.def(py::init([](uint32_t image, const py::kwargs& kw){
			Texture t(image);

			if(kw.contains("interpolation"))
				t.interpolation(kw["interpolation"].cast<std::string>());

			if(kw.contains("extension"))
				t.extension(kw["extension"].cast<std::string>());

			return t;
		}))
		.def("interpolation", &Texture::interpolation)
		.def("extension", [](Texture& self, const std::string& s){
			self.extension(s, make_float4(0));
		});


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


	m.def("cuda_sync", [](){CUDA_SYNC_CHECK();});
	m.def("linear_to_sRGB", py::vectorize(linear_to_sRGB));

}