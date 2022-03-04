#include "pt5.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <execution>
#include <algorithm>
#include <filesystem>
#include <chrono>

#include <GLFW/glfw3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


void writeImage(const std::string& filename, int w, int h, const std::vector<float>& pixels){

	{
		std::filesystem::path parent = std::filesystem::path(filename).parent_path();
		if(!( std::filesystem::exists(parent) && std::filesystem::is_directory(parent) )
			&& parent != ""){
			assert(std::filesystem::create_directory(parent));
			std::cout <<"created directory " <<parent <<std::endl;
		}
	}

	auto t0 = std::chrono::system_clock::now();

	std::vector<char> color(4*w*h);
	std::transform(std::execution::par_unseq, pixels.begin(), pixels.end(), color.begin(),
		[](float x)->char{return 255.99*pt5::linear_to_sRGB(x);});

	auto t1 = std::chrono::system_clock::now();
	std::cout <<"Time for color formatting: "
		<<std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() <<"us" <<std::endl;


	stbi_write_png(filename.c_str(), w, h, 4, color.data(), w*sizeof(uint32_t));
	std::cout <<"image saved as " <<filename <<std::endl;
}


std::shared_ptr<pt5::Scene> createScene(){
	pt5::Scene scene;

	std::shared_ptr<pt5::Image> circlesImage;
	{
		uint w = 128;
		uint h = 128;
		std::vector<float4> pixels(w*h);
		for(int i=0; i<pixels.size(); i++){
			float x = (i%w)/(float)w;
			float y = (i/w)/(float)h;
			if(x*x + y*y < 1)
				pixels[i] = make_float4(x, y, 0.5, 1);
			else
				pixels[i] = make_float4(0, x, y, 1);

			x = x*2-1;
			y = y*2-1;
			if(x*x + y*y < 6-4*1.4142135)
				pixels[i] += make_float4(1, 1, 0, 1);
		}

		circlesImage = std::make_shared<pt5::Image>(w, h, pixels);
	}


	scene.background = pt5::Material{{
		pt5::make_node(pt5::Background({{{1,0.5,1}, 1}, {1,0}})),
		pt5::make_node(pt5::Texture(circlesImage, pt5::TexType::Environment))
	}};



	scene.materials = {
		pt5::Material{{
			pt5::make_node(pt5::Diffuse({{{0.8, 0.8, 0.8}, 1}})),
			pt5::make_node(pt5::Texture(circlesImage)),
		}},

		pt5::Material{{
			pt5::make_node(pt5::Mix({1, 2, {0.5, 3}})),
			pt5::make_node(pt5::Diffuse({{{0.8, 0.2, 0.2}, 0}})),
			pt5::make_node(pt5::Diffuse({{{0.2, 0.6, 0.8}, 0}})),
			pt5::make_node(pt5::Texture(circlesImage)),
		}},

		pt5::Material{{
			pt5::make_node(pt5::Diffuse({{{0.2, 0.8, 0.2}, 0}}))
		}},

		pt5::Material{{
			pt5::make_node(pt5::Emission({{{1, 1, 1}, 0}, {10, 0}}))
		}},
	};


	{
		std::vector<pt5::Vertex> v_box = {
			// coord, normal
			{{-2, 4, 0}, { 0.5773, 0.5773, 0.5773}},
			{{-2, 0, 0}, { 0.7071, 0.0000, 0.7071}},
			{{ 2, 0, 0}, {-0.7071, 0.0000, 0.7071}},
			{{ 2, 4, 0}, {-0.5773,-0.5773, 0.5773}},

			{{-2, 4, 4}, { 0.5773,-0.5773, -0.5773}},
			{{-2, 0, 4}, { 0.7071, 0.0000, -0.7071}},
			{{ 2, 0, 4}, {-0.7071, 0.0000, -0.7071}},
			{{ 2, 4, 4}, {-0.5773,-0.5773, -0.5773}},
		};

		std::vector<float2> uv_box = {
			{0.2500, 0.0001},
			{0.2501, 0.5000},
			{0.0001, 0.2501},
			{0.5000, 0.2501},

			{0.5000, 0.7499},
			{0.7500, 0.9999},
			{0.0001, 0.7500},
			{0.9999, 0.2500},

			{0.7499, 0.5000},
			{0.9999, 0.7499},
			{0.2501, 0.9999},
			{0.7499, 0.0001},

			{0, 0},
			{0, 1},
			{1, 1},
			{1, 0}
		};

		std::vector<pt5::Face> f_box = {
			// verts, smooth, mtl
			{{0,1,2}, { 1, 2, 0}, false, 0}, {{2,3,0}, { 0, 3, 1}, false, 0}, // floor
			{{5,1,0}, {15,12,13}, false, 1}, {{0,4,5}, {13,14,15}, false, 1}, // left
			{{4,0,3}, { 4, 1, 3}, false, 0}, {{3,7,4}, { 3, 8, 4}, false, 0}, // back
			{{7,3,2}, { 8, 3,11}, false, 2}, {{2,6,7}, {11, 7, 8}, false, 2}, // right
			{{5,4,7}, { 5, 4, 8}, false, 0}, {{7,6,5}, { 8, 9, 5}, false, 0}, // roof
		};

		std::vector<uint32_t> mSlot_box = {0,1,2};


		std::vector<pt5::Vertex> v_light = {
			{{-0.5, 1.5, 3.95}, {0,0,-1}},
			{{-0.5, 2.5, 3.95}, {0,0,-1}},
			{{ 0.5, 2.5, 3.95}, {0,0,-1}},
			{{ 0.5, 1.5, 3.95}, {0,0,-1}},
		};

		std::vector<pt5::Face> f_light = {
			{{0,1,2}, {0,1,2}, false, 0},
			{{2,3,0}, {2,3,0}, false, 0},
		};

		std::vector<float2> uv_light = {
			{0, 0},
			{0, 1},
			{1, 1},
			{1, 0}
		};

		scene.meshes.emplace_back(std::make_shared<pt5::TriangleMesh>(v_box, f_box, uv_box, mSlot_box))->upload(0);
		scene.meshes.emplace_back(std::make_shared<pt5::TriangleMesh>(v_light, f_light, uv_light, std::vector<uint32_t>{3}))->upload(0);

		return std::make_shared<pt5::Scene>(scene);
	}
}


int main(int argc, char* _argv[]){
	bool background = false;
	std::string out = "result/c++.png";
	{
		std::vector<std::string> argv(_argv, _argv+argc);
		background = std::find(argv.begin(), argv.end(), "--background") != argv.end()
			|| std::find(argv.begin(), argv.end(), "-b") != argv.end();
		auto o = std::find(argv.begin(), argv.end(), "-o");
		if(o++ < argv.end()) out = *o;

		if(std::find(out.begin(), out.end(), '.') == out.end()) out += ".png";
	}

	const int width = 1024;
	const int height = 1024;

	pt5::View view(width, height);

	GLFWwindow* window = nullptr;
	if(!background && glfwInit()){
		window = glfwCreateWindow(width, height, "pt5 view", NULL, NULL);
		if(!window) glfwTerminate();
		else{
			glfwMakeContextCurrent(window);

			glEnable(GL_FRAMEBUFFER_SRGB);
			glViewport(0,0,width, height);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(0, (float)width, 0, (float)height, -1, 1);

			view.createGLTexture();
			view.clear(make_float4(0.4, 0.4, 0.4, 0.4));
		}
	}


	auto t0 = std::chrono::system_clock::now();

	auto scene = createScene();
	pt5::Camera camera{
		{0, -5, 2},
		{	{1, 0, 0},
			{0, 0,-1},
			{0, 1, 0}},
		2.3
	};

	auto t1 = std::chrono::system_clock::now();
	std::cout <<"Time for scene creation: "
		<<std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count() <<"us" <<std::endl;


	pt5::material::setNodeIndices();

	pt5::PathTracerState tracer;
	tracer.setScene(scene);

	auto t2 = std::chrono::system_clock::now();
	std::cout <<"Time for tracer construction and scene uploading: "
		<<std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count() <<"us" <<std::endl;


	tracer.render(view, 100, camera);

	while(window
		&& !glfwWindowShouldClose(window)
		&& tracer.running()
	){
		view.updateGLTexture();

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, view.GLTexture());
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


		glfwSwapBuffers(window);
    glfwPollEvents();
	};

	CUDA_SYNC_CHECK();
	auto t3 = std::chrono::system_clock::now();
	std::cout <<"Time for rendering: "
		<<std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() <<"ms" <<std::endl;

	view.downloadImage();

	writeImage(out, view.size().x, view.size().y, view.pixels);

	return 0;
}