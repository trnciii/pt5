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
	return pt5::material::Prop<float>{tu[0].cast<float>(), tu[1].cast<unsigned int>()};
}

pt5::material::Prop<float3> propf3(const py::tuple& tu){
	return pt5::material::Prop<float3>{make_float3(tu[0].cast<py::array_t<float>>()), tu[1].cast<unsigned int>()};
}



void init_material(py::module_& m){
	using namespace pt5;
	using namespace material;

	py::class_<MaterialNode, std::shared_ptr<MaterialNode>>(m, "MaterialNode");

	py::class_<Material>(m, "Material")
		.def(py::init<std::vector<std::shared_ptr<MaterialNode>>>())
		.def_readwrite("nodes", &Material::nodes);


	py::class_<Prop<float>>(m, "prop_float")
		.def(py::init<>())
		.def(py::init<float, unsigned int>())
		.def(py::init(&propf1))
		.def_readwrite("value", &Prop<float>::default_value)
		.def_readwrite("input", &Prop<float>::input);


	py::class_<Prop<float3>>(m, "prop_float3")
		.def(py::init<>())
		.def(py::init([](const py::array_t<float>& v, unsigned int i){
			return Prop<float3>{make_float3(v), i};
		}))
		.def(py::init(&propf3))
		.def_property("value", PROPERTY_FLOAT3(Prop<float3>, default_value))
		.def_readwrite("input", &Prop<float3>::input);



	py::class_<MixData>(m, "Mix")
		.def(py::init<>())
		.def(py::init<unsigned int, unsigned int, Prop<float>>())
		.def(py::init([](unsigned int b1, unsigned int b2, const py::tuple& t){
			return MixData{b1, b2, propf1(t)};
		}))
		.def(py::init([](unsigned int b1, unsigned int b2, float fv, unsigned int ft){
			return MixData({b1, b2, {fv, ft}});
		}))
		.def_readwrite("shader1", &MixData::shader1)
		.def_readwrite("shader2", &MixData::shader2)
		.def_readwrite("factor", &MixData::factor);



	py::class_<DiffuseData>(m, "Diffuse")
		.def(py::init<>())
		.def(py::init<Prop<float3>>())
		.def(py::init([](const py::tuple& t){return DiffuseData{propf3(t)};}))
		.def(py::init([](const py::array_t<float>& c, unsigned int t=0){
			return DiffuseData{{make_float3(c), t}};
		}))
		.def_readwrite("color", &DiffuseData::color);



	py::class_<EmissionData>(m, "Emission")
		.def(py::init<>())
		.def(py::init<Prop<float3>, Prop<float>>())
		.def(py::init([](const py::tuple& c, const py::tuple& s){
			return EmissionData{propf3(c), propf1(s)};
		}))
		.def(py::init([](const py::array_t<float>& c, unsigned int tc=0, float s=1, unsigned int ts=0){
			return EmissionData{{make_float3(c), tc}, {s, ts}};
		}))
		.def_readwrite("color", &EmissionData::color)
		.def_readwrite("strength", &EmissionData::strength);



	py::class_<Texture>(m, "Texture")
		.def(py::init([](uint32_t image, const py::kwargs& kw){
			cudaTextureFilterMode interpolation = cudaFilterModeLinear;
			if(kw.contains("interpolation")){
				const std::string s = kw["interpolation"].cast<std::string>();
				if(s == "Closest") interpolation = cudaFilterModePoint;
			}

			cudaTextureAddressMode extension = cudaAddressModeWrap;
			if(kw.contains("external")){
				const std::string s = kw["extension"].cast<std::string>();
				if(s == "CLIP") extension = cudaAddressModeBorder;
				else if(s == "EXTEND") extension = cudaAddressModeClamp;
			}

			return Texture(image, interpolation, extension);

		}));


	py::class_<Environment>(m, "Environment")
		.def(py::init([](uint32_t image, const py::kwargs& kw){
			cudaTextureFilterMode interpolation = cudaFilterModeLinear;
			if(kw.contains("interpolation")){
				const std::string s = kw["interpolation"].cast<std::string>();
				if(s == "Closest") interpolation = cudaFilterModePoint;
			}

			cudaTextureAddressMode extension = cudaAddressModeWrap;
			if(kw.contains("external")){
				const std::string s = kw["extension"].cast<std::string>();
				if(s == "CLIP") extension = cudaAddressModeBorder;
				else if(s == "EXTEND") extension = cudaAddressModeClamp;
			}

			return Environment(image, interpolation, extension);

		}));


	py::class_<BackgroundData>(m, "Background")
		.def(py::init<>())
		.def(py::init<Prop<float3>, Prop<float>>())
		.def(py::init([](const py::tuple& c, const py::tuple& s){
			return BackgroundData{propf3(c), propf1(s)};
		}))
		.def(py::init([](const py::array_t<float>& c, unsigned int tc=0, float s=1, unsigned int ts=0){
			return BackgroundData{{make_float3(c), tc}, {s, ts}};
		}))
		.def_readwrite("color", &BackgroundData::color)
		.def_readwrite("strength", &BackgroundData::strength);



	MAKE_NODE(MixData);
	MAKE_NODE(DiffuseData);
	MAKE_NODE(EmissionData);
	MAKE_NODE(Texture);
	MAKE_NODE(BackgroundData);
	MAKE_NODE(Environment);

}