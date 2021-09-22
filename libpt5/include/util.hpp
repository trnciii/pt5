#pragma once

#include <vector>
#include <string>
#include <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "vector_math.h"

namespace pt5{

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

} // pt5 namespace