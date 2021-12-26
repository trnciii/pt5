#pragma once

#include <memory>
#include <vector>
#include "data.h"
#include "type.hpp"
#include "../sbt.hpp"


namespace pt5{ namespace material{


inline void unwind_sbt_index(unsigned int& i, int offset_material, const std::vector<int>& offset_nodes){
	if(i>0)i = offset_material + offset_nodes[i];
}

struct DiffuseBSDF : public Node{
	DiffuseData data;

	DiffuseBSDF(const DiffuseData& d):data(d){}

	int program()const{return 0;}
	int nprograms()const{return 3;}
	MaterialNodeSBTData sbtData(int offset_material, const std::vector<int>& offset_nodes, const std::vector<cudaArray_t>& imageBuffers){
		DiffuseData ret = data;

		unwind_sbt_index(ret.color.input, offset_material, offset_nodes);

		return MaterialNodeSBTData{.diffuse = ret};
	}
};

inline std::shared_ptr<Node> make_node(const DiffuseData& data){
	return std::make_shared<DiffuseBSDF>(data);
}



struct Emission : public Node{
	EmissionData data;

	Emission(const EmissionData& d):data(d){}

	int program()const{return 3;}
	int nprograms()const{return 3;}
	MaterialNodeSBTData sbtData(int offset_material, const std::vector<int>& offset_nodes, const std::vector<cudaArray_t>& imageBuffers){
		EmissionData ret = data;
		unwind_sbt_index(ret.color.input, offset_material, offset_nodes);
		unwind_sbt_index(ret.strength.input, offset_material, offset_nodes);
		return MaterialNodeSBTData{.emission = ret};
	}

};

inline std::shared_ptr<Node> make_node(const EmissionData& data){
	return std::make_shared<Emission>(data);
}



struct Mix : public Node{
	MixData data;

	Mix(const MixData& d):data(d){}

	int program()const{return 6;}
	int nprograms()const{return 3;}
	MaterialNodeSBTData sbtData(int offset_material, const std::vector<int>& offset_nodes, const std::vector<cudaArray_t>& imageBuffers){
		MixData ret = data;

		unwind_sbt_index(ret.shader1, offset_material, offset_nodes);
		unwind_sbt_index(ret.shader2, offset_material, offset_nodes);
		unwind_sbt_index(ret.factor.input, offset_material, offset_nodes);

		return MaterialNodeSBTData{.mix = ret};
	}

};

inline std::shared_ptr<Node> make_node(const MixData& data){
	return std::make_shared<Mix>(data);
}



struct ImageTexture : public Node{
	struct CreateInfo{
		uint32_t image;
		cudaTextureDesc desc;

		CreateInfo(
			uint32_t i,
			cudaTextureFilterMode interpolation = cudaFilterModeLinear,
			cudaTextureAddressMode extension = cudaAddressModeWrap)
		:image(i){
			desc.addressMode[0] = desc.addressMode[1] = extension;
			desc.filterMode = interpolation;
			desc.normalizedCoords = 1;
			desc.maxAnisotropy = 1;
			desc.maxMipmapLevelClamp = 99;
			desc.minMipmapLevelClamp = 0;
			desc.mipmapFilterMode = cudaFilterModePoint;
		}
	};

	CreateInfo info;
	cudaTextureObject_t cudaTexture;

	ImageTexture(const CreateInfo& i):info(i){};

	~ImageTexture(){if(cudaTexture!=0)destroyCudaTexture();}

	cudaTextureObject_t createCudaTexture(const cudaArray_t& array){
		assert(cudaTexture == 0);

		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = array;
		cudaCreateTextureObject(&cudaTexture, &resDesc, &info.desc, nullptr);
		return cudaTexture;
	}

	void destroyCudaTexture(){
		assert(cudaTexture != 0);
		cudaDestroyTextureObject(cudaTexture);
		cudaTexture = 0;
	}

	int program()const{return 9;}
	int nprograms()const{return 1;}
	MaterialNodeSBTData sbtData(int offset_material, const std::vector<int>& offset_nodes, const std::vector<cudaArray_t>& imageBuffers){
		createCudaTexture(imageBuffers[info.image]);
		return MaterialNodeSBTData{.texture = cudaTexture};
	}
};

inline std::shared_ptr<Node> make_node(const ImageTexture::CreateInfo& info){
	return std::make_shared<ImageTexture>(info);
}


}}
