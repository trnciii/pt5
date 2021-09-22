#pragma once

#include <vector>
#include <string>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "vector_math.h"

namespace pt5{

void writeImage(const std::string& filename, int w, int h, const std::vector<float>& data){
	std::vector<uint32_t> pixelsi(w*h);
	for(int y=0; y<h; y++){
		for(int x=0; x<w; x++){
			const int i = w*y + x;

			const uint32_t r(255.99 * std::max(0.0f, std::min(1.0f, data[4*i  ])) );
			const uint32_t g(255.99 * std::max(0.0f, std::min(1.0f, data[4*i+1])) );
			const uint32_t b(255.99 * std::max(0.0f, std::min(1.0f, data[4*i+2])) );
			const uint32_t a(255.99 * std::max(0.0f, std::min(1.0f, data[4*i+3])) );

			pixelsi[i] = (r<<0) | (g<<8) | (b<<16) | (a<<24);
		}
	}

	stbi_write_png(filename.c_str(), w, h, 4, pixelsi.data(), w*sizeof(uint32_t));
}

} // pt5 namespace