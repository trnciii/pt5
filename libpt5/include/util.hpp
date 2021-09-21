#pragma once

#include <vector>
#include <string>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


namespace pt5{

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

} // pt5 namespace