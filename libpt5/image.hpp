#pragma once

#include <vector>

#include "vector_math.h"


namespace pt5{

struct Image{
	uint2 size;
	std::vector<float4> pixels;

	cudaArray_t array = nullptr;

	Image(uint2 s, std::vector<float4> p):size(s), pixels(p){}
	Image(uint w, uint h, std::vector<float4> p):size({w, h}), pixels(p){}

	~Image(){
		if(array != 0) cudaFreeArray(array);
	}

	void upload(){
		assert(array == 0);

		cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
		uint32_t pitch = size.x * 4 * sizeof(float);
		CUDA_CHECK(cudaMallocArray(&array, &channelFormatDesc, size.x, size.y));
		CUDA_CHECK(cudaMemcpy2DToArray(
			array,
			0, 0,
			pixels.data(),
			pitch, pitch, size.y,
			cudaMemcpyHostToDevice));
	}

	void free(){
		assert(array != 0);
		CUDA_CHECK(cudaFreeArray(array));
	}
};

}