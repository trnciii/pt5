#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cassert>

#include "pt5.hpp"
#include "./util.hpp"


namespace py = pybind11;


void init_material(py::module_& m){
	using namespace pt5;


	py::class_<BSDFData_Diffuse>(m, "BSDF_Diffuse")
		.def(py::init<>())
		.def(py::init([](const py::array_t<float>& c, uint32_t t=0){
			return BSDFData_Diffuse{{make_float3(c.at(0), c.at(1), c.at(2)), t}};
		}))
		.def_property("color",
			[](const BSDFData_Diffuse& self){return self.color.default_value;},
			[](BSDFData_Diffuse& self, const py::array_t<float>& c){
				self.color.default_value = make_float3(c.at(0), c.at(1), c.at(2));
			})
		.def_property("texture",
			[](const BSDFData_Diffuse& self){return self.color.texture;},
			[](BSDFData_Diffuse& self, uint32_t t){self.color.texture = t;});


	py::class_<BSDFData_Emission>(m, "BSDF_Emission")
		.def(py::init<>())
		.def(py::init([](const py::array_t<float>& c, uint32_t tc=0, float s=1, uint32_t ts=0){
			return BSDFData_Emission{{make_float3(c.at(0), c.at(1), c.at(2)), tc}, {s, ts}};
		}))
		.def_property("color",
			[](const BSDFData_Diffuse& self){return self.color.default_value;},
			[](BSDFData_Diffuse& self, const py::array_t<float>& c){
				self.color.default_value = make_float3(c.at(0), c.at(1), c.at(2));
			})
		.def_property("texture",
			[](const BSDFData_Diffuse& self){return self.color.texture;},
			[](BSDFData_Diffuse& self, uint32_t t){self.color.texture = t;});


	py::class_<Material>(m, "Material")
		.def(py::init([](const py::list& li){
			std::vector<std::shared_ptr<MaterialNode>> nodes;
			for(const py::handle& obj : li){
				if(py::isinstance<BSDFData_Diffuse>(obj))
					nodes.push_back(make_node(obj.cast<BSDFData_Diffuse>()));
				else if(py::isinstance<BSDFData_Emission>(obj))
					nodes.push_back(make_node(obj.cast<BSDFData_Emission>()));
				else std::cout <<"error: " <<obj <<std::endl;
			}
			return Material{nodes};
		}))
		.def_property("nodes",
		[](const Material& self){
			py::list li;
			for(const std::shared_ptr<MaterialNode>& m: self.nodes){
				if(m->type() == MaterialType::Diffuse)
					li.append(*(BSDFData_Diffuse*)m->ptr());
				else if(m->type() == MaterialType::Emission)
					li.append(*(BSDFData_Emission*)m->ptr());
				else
					li.append("error");
			}
			return li;
		},
		[](Material& self, const py::list& li){
			std::vector<std::shared_ptr<MaterialNode>> nodes;
			for(const py::handle& obj : li){
				if(py::isinstance<BSDFData_Diffuse>(obj))
					nodes.push_back(make_node(obj.cast<BSDFData_Diffuse>()));
				else if(py::isinstance<BSDFData_Emission>(obj))
					nodes.push_back(make_node(obj.cast<BSDFData_Emission>()));
				else std::cout <<"error: " <<obj <<std::endl;
			}
			self.nodes = nodes;
		});


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


}