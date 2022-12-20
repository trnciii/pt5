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

	m.def("setNodeIndices", &setNodeIndices);


	py::class_<MaterialNode, std::shared_ptr<MaterialNode>>(m, "MaterialNode");

	py::class_<Material>(m, "Material")
		.def(py::init<std::vector<std::shared_ptr<MaterialNode>>>())
		.def_readwrite("nodes", &Material::nodes);



	py::class_<MixData>(m, "Mix")
		.def(py::init<>())
		.def(py::init([](unsigned int b1, unsigned int b2, const py::tuple& t){
			return MixData{b1, b2, propf1(t)};
		}))
		.def_readwrite("shader1", &MixData::shader1)
		.def_readwrite("shader2", &MixData::shader2)
		.def_readwrite("factor", &MixData::factor);



	py::class_<DiffuseData>(m, "Diffuse")
		.def(py::init<>())
		.def(py::init([](const py::tuple& t){return DiffuseData{propf3(t)};}))
		.def_readwrite("color", &DiffuseData::color);


	py::class_<GlossyData>(m, "Glossy")
		.def(py::init<>())
		.def(py::init([](const py::tuple& c, const py::tuple& r){return GlossyData{propf3(c), propf1(r)};}))
		.def_readwrite("color", &GlossyData::color)
		.def_readwrite("alpha", &GlossyData::alpha);


	py::class_<MeasuredG1>(m, "MeasuredG1")
		.def(py::init<>())
		.def(py::init([](const py::tuple& c, const py::tuple& a, const py::array_t<float>& t){
			assert(t.ndim() == 2);
			return MeasuredG1({
				propf3(c), propf1(a),
				std::vector<float>((float*)t.data(), (float*)t.data() + t.size()),
				{t.shape(0), t.shape(1)}});
		}))
		.def_readwrite("color", &MeasuredG1::color)
		.def_readwrite("alpha", &MeasuredG1::alpha)
		.def_property("table",
			[](const MeasuredG1& self){
				return (py::array_t<float>)py::buffer_info(
					(float*)self.table.data(),
					sizeof(float),
					py::format_descriptor<float>::format(),
					2, {self.shape.x, self.shape.y}, {self.shape.y*sizeof(float), sizeof(float)}
				);
			},
			[](MeasuredG1& self, py::array_t<float>& t){
				assert(t.ndim() == 2);
				self.table = std::vector<float>((float*)t.data(), (float*)t.data() + t.size());
				self.shape = {t.shape(0), t.shape(1)};
			}
		);



	py::class_<EmissionData>(m, "Emission")
		.def(py::init<>())
		.def(py::init([](const py::tuple& c, const py::tuple& s){
			return EmissionData{propf3(c), propf1(s)};
		}))
		.def_readwrite("color", &EmissionData::color)
		.def_readwrite("strength", &EmissionData::strength);



	py::class_<Texture>(m, "Texture")
		.def(py::init([](std::shared_ptr<Image> image, const py::kwargs& kw){
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

			TexType type = TexType::ImageTexture;
			if(kw.contains("type")){
				const std::string s = kw["type"].cast<std::string>();
				if(s == "TEX_ENVIRONMENT")
					type = TexType::Environment;
			}

			return Texture(image, type, interpolation, extension);
		}))
		.def_readwrite("image", &Texture::image);


	py::class_<BackgroundData>(m, "Background")
		.def(py::init<>())
		.def(py::init([](const py::tuple& c, const py::tuple& s){
			return BackgroundData{propf3(c), propf1(s)};
		}))
		.def_readwrite("color", &BackgroundData::color)
		.def_readwrite("strength", &BackgroundData::strength);



	MAKE_NODE(MixData);
	MAKE_NODE(DiffuseData);
	MAKE_NODE(GlossyData);
	MAKE_NODE(MeasuredG1);
	MAKE_NODE(EmissionData);
	MAKE_NODE(Texture);
	MAKE_NODE(BackgroundData);

}