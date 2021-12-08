#pragma once

#include "optix.hpp"
#include <iostream>

namespace pt5{

struct Texture{
	uint2 size;
	std::vector<float4> pixels;
};


class CUDATexture{
	cudaArray_t cudaTextureData {};
	cudaTextureObject_t cudaTexture {};

public:
	void upload(const Texture& texture){
		cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();

		uint32_t pitch = texture.size.x * 4 * sizeof(float);
		CUDA_CHECK(cudaMallocArray(&cudaTextureData, &channelFormatDesc, texture.size.x, texture.size.y));
		CUDA_CHECK(cudaMemcpy2DToArray(
			cudaTextureData,
			0, 0,
			texture.pixels.data(),
			pitch, pitch, texture.size.y,
			cudaMemcpyHostToDevice));


		cudaResourceDesc resDesc = {};
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cudaTextureData;

		cudaTextureDesc textureDesc = {};
			textureDesc.addressMode[0] = cudaAddressModeWrap;
			textureDesc.addressMode[1] = cudaAddressModeWrap;
			textureDesc.filterMode = cudaFilterModeLinear;
			// textureDesc.readMode = cudaReadModeNormalizedFloat;
			textureDesc.normalizedCoords = 1;
			textureDesc.maxAnisotropy = 1;
			textureDesc.maxMipmapLevelClamp = 99;
			textureDesc.minMipmapLevelClamp = 0;
			textureDesc.mipmapFilterMode = cudaFilterModePoint;
			textureDesc.borderColor[0] = 1.0f;
			textureDesc.sRGB = 2;

		CUDA_CHECK(cudaCreateTextureObject(&cudaTexture, &resDesc, &textureDesc, nullptr));
	}

	void free(){
		CUDA_CHECK(cudaFreeArray(cudaTextureData));
		CUDA_CHECK(cudaDestroyTextureObject(cudaTexture));
	}

	cudaTextureObject_t id()const {return cudaTexture;}
};

}