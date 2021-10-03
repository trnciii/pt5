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
	camera.position = {0, -5, 2};
	camera.toWorld[0] = {1, 0, 0};
	camera.toWorld[1] = {0, 0,-1};
	camera.toWorld[2] = {0, 1, 0};
	camera.focalLength = 2.3;


	scene.background = make_float3(0.2, 0, 0.4);


	scene.materials = {
		// albedo, emission
		{{0.8, 0.8, 0.8}, {0, 0, 0}}, // white
		{{0.8, 0.2, 0.2}, {0, 0, 0}},	// red
		{{0.2, 0.8, 0.2}, {0, 0, 0}}, // green
		{{  0,   0,   0}, {10, 10, 10}}  // light
	};


	std::vector<float3> v_box = {
		{-2, 4, 0},
		{-2, 0, 0},
		{ 2, 0, 0},
		{ 2, 4, 0},

		{-2, 4, 4},
		{-2, 0, 4},
		{ 2, 0, 4},
		{ 2, 4, 4},
	};

	std::vector<float3> n_box = {
		{ 0.5773, 0.5773, 0.5773},
		{ 0.7071, 0.0000, 0.7071},
		{-0.7071, 0.0000, 0.7071},
		{-0.5773, -0.5773, 0.5773},

		{ 0.5773,-0.5773, -0.5773},
		{ 0.7071, 0.0000, -0.7071},
		{-0.7071, 0.0000, -0.7071},
		{-0.5773,-0.5773, -0.5773}
	};

	std::vector<uint3> f_box = {
		{0,1,2}, {2,3,0}, // floor
		{5,1,0}, {0,4,5}, // left
		{4,0,3}, {3,7,4}, // back
		{7,3,2}, {2,6,7}, // right
		{5,4,7}, {7,6,5}, // roof
	};

	std::vector<uint32_t> mSlot_box = {0,1,2};

	std::vector<uint32_t> mIndex_box = {
		0, 0,
		1, 1,
		0, 0,
		2, 2,
		0, 0
	};


	std::vector<float3> v_light = {
		{-0.5, 1.5, 3.95},
		{-0.5, 2.5, 3.95},
		{ 0.5, 2.5, 3.95},
		{ 0.5, 1.5, 3.95},
	};

	std::vector<float3> n_light = {
		{0, 0, -1},
		{0, 0, -1},
		{0, 0, -1},
		{0, 0, -1}
	};

	std::vector<uint3> f_light = {{0,1,2}, {2,3,0}};
	std::vector<uint32_t> mSlot_light = {3};
	std::vector<uint32_t> mIndex_light = {0, 0};


	scene.meshes.push_back(pt5::TriangleMesh{v_box, n_box, f_box, mIndex_box, mSlot_box});
	scene.meshes.push_back(pt5::TriangleMesh{v_light, n_light, f_light, mIndex_light, mSlot_light});
}


int main(){

	const int width = 1024;
	const int height = 1024;

	pt5::Scene scene;
	createScene(scene);

	pt5::PathTracerState tracer;
	tracer.init();
	tracer.setScene(scene);
	tracer.initLaunchParams(width, height, 1000);

	tracer.render();

	std::vector<float> pixels = tracer.pixels();
	writeImage("out_c++.png", width, height, pixels);
	std::cout <<"image saved" <<std::endl;

	return 0;
}