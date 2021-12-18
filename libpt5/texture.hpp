#pragma once

#include <iostream>
#include "optix.hpp"

namespace pt5{

struct Texture{
	uint2 size;
	std::vector<float4> pixels;
	cudaTextureDesc desc;

	Texture(const uint2& s, const std::vector<float4>& p)
	:size(s), pixels(p){
		desc.addressMode[0] = desc.addressMode[1] = cudaAddressModeWrap;
		desc.filterMode = cudaFilterModeLinear;
		desc.normalizedCoords = 1;
		desc.maxAnisotropy = 1;
		desc.maxMipmapLevelClamp = 99;
		desc.minMipmapLevelClamp = 0;
		desc.mipmapFilterMode = cudaFilterModePoint;
	}

	void interpolation(const std::string& s){
		if(s == "Linear")
			desc.filterMode = cudaFilterModeLinear;
		else if(s == "Closest")
			desc.filterMode = cudaFilterModePoint;
		else
			std::cout <<s <<" not found in ('Linear', 'Closest')." <<std::endl;
	}

	void extension(const std::string& s, float4 c = {0,0,0,0}){
		if(s == "REPEAT")
			desc.addressMode[0] = desc.addressMode[1] = cudaAddressModeWrap;
		else if(s == "CLIP"){
			desc.addressMode[0] = desc.addressMode[1] = cudaAddressModeBorder;
			desc.borderColor[0] = c.x;
			desc.borderColor[1] = c.y;
			desc.borderColor[2] = c.z;
			desc.borderColor[3] = c.w;
		}
		else if(s == "EXTEND")
			desc.addressMode[0] = desc.addressMode[1] = cudaAddressModeClamp;
		else
			std::cout <<s <<" not found in ('REPEAT', 'EXTEND', 'CLIP')" <<std::endl;
	}

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


		CUDA_CHECK(cudaCreateTextureObject(&cudaTexture, &resDesc, &texture.desc, nullptr));
	}

	void free(){
		CUDA_CHECK(cudaFreeArray(cudaTextureData));
		CUDA_CHECK(cudaDestroyTextureObject(cudaTexture));
	}

	cudaTextureObject_t id()const {return cudaTexture;}
};

}