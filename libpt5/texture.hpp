#pragma once

#include <iostream>
#include "optix.hpp"

namespace pt5{

struct Image{
	uint2 size;
	std::vector<float4> pixels;
};

struct Texture{
	uint32_t image;
	cudaTextureDesc desc;

	Texture(uint32_t i):image(i){
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


}