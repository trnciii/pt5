#include "pt5.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <filesystem>

#include <GLFW/glfw3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


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

class Viewer{
public:
	Viewer(int w, int h, pt5::CUDABuffer& buf):width(w), height(h), buffer(buf){
		CUDA_CHECK(cudaStreamCreate(&stream));
		pixels.resize(4*width*height);
	}

	~Viewer(){}

	void downloadImage(){
		buffer.download(pixels.data(), pixels.size(), stream);
	}

	void initWindow(){
		if(!glfwInit()) assert(0);

		window = glfwCreateWindow(width, height, "pt5 view", NULL, NULL);
		if (!window){
				assert(0);
		    glfwTerminate();
		}

		glfwMakeContextCurrent(window);


		glDisable(GL_LIGHTING);
		glDisable(GL_DEPTH_TEST);

		glViewport(0,0,width, height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, (float)width, 0, (float)height, -1, 1);
	}

	void draw(){
		downloadImage();

		GLuint fbTexture {0};
		glGenTextures(1, &fbTexture);
		glBindTexture(GL_TEXTURE_2D, fbTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, pixels.data());

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, fbTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.f, 0.f);
			glVertex3f(0.f, (float)height, 0.f);

			glTexCoord2f(0.f, 1.f);
			glVertex3f(0.f, 0.f, 0.f);

			glTexCoord2f(1.f, 1.f);
			glVertex3f((float)width, 0.f, 0.f);

			glTexCoord2f(1.f, 0.f);
			glVertex3f((float)width, (float)height, 0.f);
		}
		glEnd();

	}


	void run(const cudaEvent_t* const finished){
		do{
			draw();
			glfwSwapBuffers(window);
	    glfwPollEvents();
		}while(
			!glfwWindowShouldClose(window)
			&& (cudaEventQuery(*finished) == cudaErrorNotReady)
			);
	}


	CUstream stream;
	pt5::CUDABuffer& buffer;
	int width;
	int height;
	std::vector<float> pixels;
	GLFWwindow* window;
};


int main(){

	const int width = 1024;
	const int height = 1024;

	pt5::Scene scene;
	createScene(scene);

	pt5::PathTracerState tracer;
	tracer.init();
	tracer.setScene(scene);
	tracer.initLaunchParams(width, height, 1000);

	Viewer viewer(width, height, tracer.pixelBuffer);
	viewer.initWindow();


	tracer.render();
	viewer.run(&tracer.finished);

	CUDA_SYNC_CHECK();


	viewer.downloadImage();

	std::string outDir("result");
	if(!( std::filesystem::exists(outDir) && std::filesystem::is_directory(outDir) )){
		std::cout <<"created directory " <<outDir <<std::endl;
		assert(std::filesystem::create_directory(outDir));
	}

	writeImage(outDir+"/out_c++.png", width, height, viewer.pixels);
	std::cout <<"image saved" <<std::endl;

	return 0;
}