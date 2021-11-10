#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#define GLAD_GL_IMPLEMENTATION
#include <glad/gl.h>
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
		auto r = x.unchecked<1>();                       \
		self.member = make_float3(r(0), r(1), r(2));     \
	}


template <typename T>
std::vector<T> toSTDVector(py::array_t<T>& x){
	return std::vector<T>(x.data(0), x.data(0)+x.size());
}


class Window{
public:
	Window(pt5::View& v):view(v){
		const int x = view.size().x;
		const int y = view.size().y;

		if(!glfwInit()) return;

		window = glfwCreateWindow(x, y, "pt5 view", NULL, NULL);
		if(!window){
			glfwTerminate();
			return;
		}

		glfwMakeContextCurrent(window);
		gladLoadGL(glfwGetProcAddress);

		glEnable(GL_FRAMEBUFFER_SRGB);
		glViewport(0,0,x,y);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, (float)x, 0, (float)y, -1, 1);

		initPrograms();
		createVAO();

		view.createGLTexture();

		std::cout <<"created Window" <<std::endl;
	}

	~Window(){
		if(!window) return;

		view.destroyGLTexture(); // destruct view within opengl context.
		glDeleteBuffers(1, &vertexBuffer);
		glDeleteBuffers(1, &indexBuffer);
		glDeleteVertexArrays(1, &vertexArray);
		glDeleteProgram(program);
		glfwDestroyWindow(window);
		glfwTerminate();

		std::cout <<"destroyed Window" <<std::endl;
	}


	void draw(const pt5::PathTracerState& tracer){
		if(!window){
			CUDA_SYNC_CHECK();
			return;
		}

	  glUniform1i(glGetUniformLocation(program, "txBuffer"), 0);

		glUseProgram(program);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, view.GLTexture());
		glBindVertexArray(vertexArray);


		while(!glfwWindowShouldClose(window)
			&& tracer.running()
		){
			view.updateGLTexture();

			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

			glfwSwapBuffers(window);
	    glfwPollEvents();
		};

		CUDA_SYNC_CHECK();
	}

	bool hasWindow(){return window;}

private:
	GLuint compileShader(const std::string& source, const GLuint type){
		const GLchar* source_data = (GLchar*)source.c_str();

		GLuint shader = glCreateShader(type);
		glShaderSource(shader, 1, &source_data, nullptr);
		glCompileShader(shader);

		GLint compiled = GL_FALSE;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

		return shader;
	}

	void initPrograms(){
		{
			std::string source =
				"#version 460\n"

				"layout (location = 0) in vec3 position;\n"
				"layout (location = 1) in vec2 txCoord;\n"

				"out vec2 co;\n"

				"void main(){\n"
				"		co = txCoord;\n"
				"		gl_Position = vec4( position, 1.0 );\n"
				"}\n";

			vs = compileShader(source, GL_VERTEX_SHADER);
		}
		{
			std::string source =
				"#version 460\n"

				"in vec2 co;\n"
				"out vec4 fragColor;\n"

				"uniform sampler2D txBuffer;\n"

				"void main(){\n"
				"		fragColor = texture(txBuffer, co);\n"
				"}\n";

			fs = compileShader(source, GL_FRAGMENT_SHADER);
		}

		program = glCreateProgram();
		glAttachShader(program, vs);
		glAttachShader(program, fs);
		glLinkProgram(program);

		glDetachShader(program, vs);
		glDetachShader(program, fs);
		glDeleteShader(vs);
		glDeleteShader(fs);
	}

	void createVAO(){
		GLfloat vertices[] = {
			-1, 1, 0,  0, 0,
			 1, 1, 0,  1, 0,
			 1,-1, 0,	 1, 1,
			-1,-1, 0,  0, 1
		};

		GLuint indices[] = {
			0, 1, 2,
			2, 3, 0
		};

		glGenBuffers(1, &vertexBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glGenVertexArrays(1, &vertexArray);
		glBindVertexArray(vertexArray);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), 0);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(GLfloat), (void*)(3*sizeof(GLfloat)));
		glEnableVertexAttribArray(1);


		glGenBuffers(1, &indexBuffer);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	}


	GLFWwindow* window;
	pt5::View& view;

	GLuint vs;
	GLuint fs;
	GLuint program;

	GLuint vertexBuffer;
	GLuint vertexArray;
	GLuint indexBuffer;
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

	py::class_<Window>(m, "Window_cpp")
		.def(py::init<View&>())
		.def("draw", &Window::draw)
		.def_property_readonly("hasWindow", &Window::hasWindow);


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


	py::class_<TriangleMesh>(m, "TriangleMesh")
		.def(py::init([](
			py::array_t<float>& v,
			py::array_t<float>& n,
			py::array_t<uint32_t>& f,
			py::array_t<bool>& smooth,
			py::array_t<uint32_t>& mIdx,
			py::array_t<uint32_t>& mSlt)
		{
			return TriangleMesh(
				std::vector<float3>( (float3*)v.data(0,0), (float3*)v.data(0,0)+v.shape(0) ),
				std::vector<float3>( (float3*)n.data(0,0), (float3*)n.data(0,0)+n.shape(0) ),
				std::vector<uint3>( (uint3*)f.data(0,0), (uint3*)f.data(0,0)+f.shape(0)),
				toSTDVector(smooth),
				toSTDVector(mIdx),
				(mSlt.size()>0)? toSTDVector(mSlt) : std::vector<uint32_t>(0));
		}));

	m.def("cuda_sync", [](){CUDA_SYNC_CHECK();});

}