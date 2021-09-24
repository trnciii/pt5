#include "pt5.hpp"
#include "util.hpp"

#include <vector>
#include <string>
#include <iostream>


void createScene(pt5::Scene& scene){
	pt5::Camera& camera = scene.camera;
	camera.position = {0, -40, 1};
	camera.toWorld[0] = {1, 0, 0};
	camera.toWorld[1] = {0, 0,-1};
	camera.toWorld[2] = {0, 1, 0};
	camera.focalLength = 2;


	scene.background = make_float3(0.5, 0.2, 0.8);


	std::vector<pt5::Vertex> v0 = {
		{{-4, 0, 6}, {0, -1, 0}},
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
		{{0, 1, 2}},
		{{2, 3, 0}},
		{{3, 2, 5}},
		{{5, 4, 3}}
	};


	std::vector<pt5::Face> f1 = {
		{{0, 1, 2}},
		{{2, 3, 0}},
		{{3, 2, 5}},
		{{5, 4, 3}}
	};

	scene.meshes.push_back(pt5::TriangleMesh{v0, f0});
	scene.meshes.push_back(pt5::TriangleMesh{v1, f1});
}


int main(){

	const int width = 1200;
	const int height = 800;

	pt5::Scene scene;
	createScene(scene);

	pt5::PathTracerState tracer;
	tracer.init();
	tracer.setScene(scene);
	tracer.initLaunchParams(width, height);

	tracer.render();

	std::vector<float> pixels = tracer.pixels();
	pt5::writeImage("out_c++.png", width, height, pixels);
	std::cout <<"image saved" <<std::endl;

	return 0;
}