#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cassert>

#include "pt5.hpp"
#include "./util.hpp"


namespace py = pybind11;


#define MAKE_NODE(type){                                                           \
  m.def("make_node", (std::shared_ptr<MaterialNode> (*)(const type&)) &make_node); \
}


pt5::material::Prop<float> propf1(const py::tuple& tu){
	return pt5::material::Prop<float>{tu[0].cast<float>(), tu[1].cast<int>()};
}

pt5::material::Prop<float3> propf3(const py::tuple& tu){
	return pt5::material::Prop<float3>{make_float3(tu[0].cast<py::array_t<float>>()), tu[1].cast<int>()};
}



void init_material(py::module_& m){
	using namespace pt5;

	py::class_<MaterialNode, std::shared_ptr<MaterialNode>>(m, "MaterialNode");

	py::class_<Material>(m, "Material")
		.def(py::init<std::vector<std::shared_ptr<MaterialNode>>>())
		.def_readwrite("nodes", &Material::nodes);


	using material::Prop;

	py::class_<Prop<float>>(m, "prop_float")
		.def(py::init<>())
		.def(py::init<float, int>())
		.def(py::init(&propf1))
		.def_readwrite("value", &Prop<float>::default_value)
		.def_readwrite("input", &Prop<float>::texture);


	py::class_<Prop<float3>>(m, "prop_float3")
		.def(py::init<>())
		.def(py::init([](const py::array_t<float>& v, int i){
			return Prop<float3>{make_float3(v), i};
		}))
		.def(py::init(&propf3))
		.def_property("value", PROPERTY_FLOAT3(Prop<float3>, default_value))
		.def_readwrite("input", &Prop<float3>::texture);



	py::class_<BSDFData_Mix>(m, "BSDF_Mix")
		.def(py::init<>())
		.def(py::init<int, int, Prop<float>>())
		.def(py::init([](int b1, int b2, const py::tuple& t){
			return BSDFData_Mix{b1, b2, propf1(t)};
		}))
		.def(py::init([](int b1, int b2, float fv, int ft){
			return BSDFData_Mix({b1, b2, {fv, ft}});
		}))
		.def_readwrite("bsdf1", &BSDFData_Mix::bsdf1)
		.def_readwrite("bsdf2", &BSDFData_Mix::bsdf2)
		.def_readwrite("factor", &BSDFData_Mix::factor);



	py::class_<BSDFData_Diffuse>(m, "BSDF_Diffuse")
		.def(py::init<>())
		.def(py::init<Prop<float3>>())
		.def(py::init([](const py::tuple& t){return BSDFData_Diffuse{propf3(t)};}))
		.def(py::init([](const py::array_t<float>& c, int t=0){
			return BSDFData_Diffuse{{make_float3(c), t}};
		}))
		.def_readwrite("color", &BSDFData_Diffuse::color);



	py::class_<BSDFData_Emission>(m, "BSDF_Emission")
		.def(py::init<>())
		.def(py::init<Prop<float3>, Prop<float>>())
		.def(py::init([](const py::tuple& c, const py::tuple& s){
			return BSDFData_Emission{propf3(c), propf1(s)};
		}))
		.def(py::init([](const py::array_t<float>& c, int tc=0, float s=1, int ts=0){
			return BSDFData_Emission{{make_float3(c), tc}, {s, ts}};
		}))
		.def_readwrite("color", &BSDFData_Emission::color)
		.def_readwrite("strength", &BSDFData_Emission::strength);



	MAKE_NODE(BSDFData_Mix);
	MAKE_NODE(BSDFData_Diffuse);
	MAKE_NODE(BSDFData_Emission);



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