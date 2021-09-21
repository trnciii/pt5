#include "pt5.hpp"

#include <vector>
#include <string>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "vector_math.h"


void writeImage(const std::string& filename, int w, int h, const std::vector<float>& data){
	std::vector<uint32_t> pixelsi(w*h);
	for(int y=0; y<h; y++){
		for(int x=0; x<w; x++){

			int i = w*y + x;

			uint32_t r(data[4*i  ]*255.99);
			uint32_t g(data[4*i+1]*255.99);
			uint32_t b(data[4*i+2]*255.99);
			uint32_t a(data[4*i+3]*255.99);

			pixelsi[i] = (r<<0) | (g<<8) | (b<<16) | (a<<24);
		}
	}

	stbi_write_png(filename.c_str(), w, h, 4, pixelsi.data(), w*sizeof(uint32_t));
}

int main(){
	int width = 1200;
	int height = 800;

	std::vector<float> pixelsf(4*width*height);
	for(int y=0; y<height; y++){
		for(int x=0; x<width; x++){

			const int i = width*y + x;

			float r = float(x%256)/256;
			float g = float(y%256)/256;
			float b = 0.5;

			pixelsf[4*i  ] = r;
			pixelsf[4*i+1] = g;
			pixelsf[4*i+2] = b;
			pixelsf[4*i+3] = 1;
		}
	}

	writeImage("out.png", width, height, pixelsf);


	return 0;
}