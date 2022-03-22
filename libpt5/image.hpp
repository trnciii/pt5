#pragma once

#include <vector>

#include "vector_math.h"


namespace pt5{

struct Image{
	uint2 size = {0,0};
	cudaArray_t array = nullptr;

	~Image(){
		if(array != 0) cudaFreeArray(array);
	}

	void alloc(uint w, uint h){
		assert(array == nullptr);
		cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();
		CUDA_CHECK(cudaMallocArray(&array, &channelFormatDesc, w, h));
		size = {w, h};
	}

	void upload(const std::vector<float4>& pixels, CUstream stream){
		assert(pixels.size() == (size.x*size.y));
		upload(pixels.data(), stream);
	}

	void upload(const float4* data, CUstream stream){
		assert(array != nullptr);

		uint32_t pitch = size.x * sizeof(float4);
		CUDA_CHECK(cudaMemcpy2DToArrayAsync(
			array,
			0, 0,
			data,
			pitch, pitch, size.y,
			cudaMemcpyDefault,
			stream
		));
	}

	void alloc_and_upload(uint w, uint h, const std::vector<float4>& pixels, CUstream stream){
		alloc(w, h);
		upload(pixels, stream);
	}

	void alloc_and_upload(uint w, uint h, const float4* data, CUstream stream){
		alloc(w, h);
		upload(data, stream);
	}

	void free(){
		assert(array != nullptr);
		CUDA_CHECK(cudaFreeArray(array));
		size = {0,0};
	}
};

}