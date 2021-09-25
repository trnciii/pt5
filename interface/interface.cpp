#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pt5.hpp"


void createScene(pt5::Scene& scene){
	pt5::Camera& camera = scene.camera;
	camera.position = {0, -20, 1};
	camera.toWorld[0] = {1, 0, 0};
	camera.toWorld[1] = {0, 0,-1};
	camera.toWorld[2] = {0, 1, 0};
	camera.focalLength = 2;


	scene.background = make_float3(0.4, 0.1, 0.8);


	scene.materials = {
		{{0.8, 0.2, 0.2}}, // color
		{{0.2, 0.8, 0.2}},
		{{0.2, 0.2, 0.8}},
		{{0.8, 0.8, 0.4}}
	};


	std::vector<pt5::Vertex> v0 = {
		{{-4, 0, 6}, {0, -1, 0}}, // position, normal
		{{-4, 0, 2}, {0, -1, 0}},
		{{ 0, 0, 2}, {0, -1, 0}},
		{{ 0, 0, 6}, {0, -1, 0}},
		{{ 4, 0, 6}, {0, -1, 0}},
		{{ 4, 0, 2}, {0, -1, 0}}
	};

	std::vector<pt5::Vertex> v1 = {
		{{-4, 0, 6-6}, {0, -1, 0}},
		{{-4, 0, 2-6}, {0, -1, 0}},
		{{ 0, 0, 2-6}, {0, -1, 0}},
		{{ 0, 0, 6-6}, {0, -1, 0}},
		{{ 4, 0, 6-6}, {0, -1, 0}},
		{{ 4, 0, 2-6}, {0, -1, 0}}
	};


	std::vector<pt5::Face> f0 = {
		{{0, 1, 2}, 0}, // indices, material
		{{2, 3, 0}, 1},
		{{3, 2, 5}, 2},
		{{5, 4, 3}, 3}
	};


	std::vector<pt5::Face> f1 = {
		{{0, 1, 2}, 1},
		{{2, 3, 0}, 1},
		{{3, 2, 5}, 0},
		{{5, 4, 3}, 0}
	};

	scene.meshes.push_back(pt5::TriangleMesh{v0, f0});
	scene.meshes.push_back(pt5::TriangleMesh{v1, f1});
}



namespace py = pybind11;

PYBIND11_MODULE(core, m) {
	using namespace pt5;

	py::class_<PathTracerState>(m, "PathTracer")
		.def(py::init<>())
		.def("init", &PathTracerState::init)
		.def("setScene", &PathTracerState::setScene)
		.def("initLaunchParams", &PathTracerState::initLaunchParams)
		.def("render", &PathTracerState::render)
		.def("pixels", &PathTracerState::pixels);


	py::class_<Scene>(m, "Scene")
		.def(py::init<>())
		.def("createDefault", &createScene);

}