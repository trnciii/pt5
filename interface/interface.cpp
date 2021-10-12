#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <GLFW/glfw3.h>

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


void cuda_sync(){
	CUDA_SYNC_CHECK();
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


class Window{
public:
	Window(int x, int y){
		if(!glfwInit()) assert(0);

		window = glfwCreateWindow(x, y, "pt5 view", NULL, NULL);
		if (!window){
				assert(0);
		    glfwTerminate();
		}

		glfwMakeContextCurrent(window);

		glEnable(GL_FRAMEBUFFER_SRGB);
		glViewport(0,0,x,y);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, (float)x, 0, (float)y, -1, 1);

		glGenTextures(1, &tx);
	}


	void draw(pt5::View& view, pt5::PathTracerState& tracer){
		glfwSetWindowSize(window, view.size().x, view.size().y);

		do{
			view.updateGLTexture();

			glEnable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, tx);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

			glBegin(GL_QUADS);
			{
				glTexCoord2f(0.f, 0.f);
				glVertex3f(0.f, (float)view.size().y, 0.f);

				glTexCoord2f(0.f, 1.f);
				glVertex3f(0.f, 0.f, 0.f);

				glTexCoord2f(1.f, 1.f);
				glVertex3f((float)view.size().x, 0.f, 0.f);

				glTexCoord2f(1.f, 0.f);
				glVertex3f((float)view.size().x, (float)view.size().y, 0.f);
			}
			glEnd();


			glfwSwapBuffers(window);
	    glfwPollEvents();
		}while(
			!glfwWindowShouldClose(window)
			&& tracer.running()
		);
	}

	GLuint texture(){return tx;}

private:
	GLFWwindow* window;
	GLuint tx;
};



PYBIND11_MODULE(core, m) {
	using namespace pt5;

	py::class_<View>(m, "View")
		.def(py::init<int, int>())
		.def("downloadImage", &View::downloadImage)
		.def("registerGLTexture", &View::registerGLTexture)
		.def("updateGLTexture", &View::updateGLTexture)
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

	py::class_<Window>(m, "Window")
		.def(py::init<int, int>())
		.def("draw", &Window::draw)
		.def_property_readonly("texture", &Window::texture);


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