#include "pt5.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "vector_math.h"


void writeImage(const std::string& filename, int w, int h, const std::vector<float>& color_float){
	std::vector<uint32_t> color(w*h);
	for(int i=0; i<w*h; i++){
		const uint32_t r(255.99 * std::max(0.0f, std::min(1.0f, color_float[4*i  ])) );
		const uint32_t g(255.99 * std::max(0.0f, std::min(1.0f, color_float[4*i+1])) );
		const uint32_t b(255.99 * std::max(0.0f, std::min(1.0f, color_float[4*i+2])) );
		const uint32_t a(255.99 * std::max(0.0f, std::min(1.0f, color_float[4*i+3])) );

		color[i] = (r<<0) | (g<<8) | (b<<16) | (a<<24);
	}

	stbi_write_png(filename.c_str(), w, h, 4, color.data(), w*sizeof(uint32_t));
}


void createScene(pt5::Scene& scene){
	pt5::Camera& camera = scene.camera;
	camera.position = {0, -20, 1};
	camera.toWorld[0] = {1, 0, 0};
	camera.toWorld[1] = {0, 0,-1};
	camera.toWorld[2] = {0, 1, 0};
	camera.focalLength = 2;


	scene.background = make_float3(0.2, 0, 0.4);


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
		{{0, 1, 2}, 1}, // indices, material
		{{2, 3, 0}, 1},
		{{3, 2, 5}, 0},
		{{5, 4, 3}, 0}
	};


	std::vector<pt5::Face> f1 = {
		{{0, 1, 2}, 1},
		{{2, 3, 0}, 2},
		{{3, 2, 5}, 0},
		{{5, 4, 3}, 1}
	};


	scene.meshes.push_back(pt5::TriangleMesh{v0, f0, {3, 0}});
	scene.meshes.push_back(pt5::TriangleMesh{v1, f1, {1, 0, 2}});
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
	writeImage("out_c++.png", width, height, pixels);
	std::cout <<"image saved" <<std::endl;

	return 0;
}