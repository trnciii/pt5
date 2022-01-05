#pragma once

#include <memory>
#include <vector>
#include "data.h"
#include "type.hpp"
#include "../sbt.hpp"

#include <iostream>

namespace pt5{ namespace material{

class DiffuseBSDF : public Node{
	DiffuseData data;

public:
	DiffuseBSDF(const DiffuseData& d):data(d){}

	int program()const{return 0;}
	int nprograms()const{return 3;}
	MaterialNodeSBTData sbtData(const NodeIndexingInfo& i){
		DiffuseData ret = data;

		ret.color.input = i.index_node(data.color.input);

		return MaterialNodeSBTData{.diffuse = ret};
	}
};

inline std::shared_ptr<Node> make_node(const DiffuseData& data){
	return std::make_shared<DiffuseBSDF>(data);
}



class Emission : public Node{
	EmissionData data;

public:
	Emission(const EmissionData& d):data(d){}

	int program()const{return 3;}
	int nprograms()const{return 3;}
	MaterialNodeSBTData sbtData(const NodeIndexingInfo& i){
		EmissionData ret = data;

		ret.color.input = i.index_node(data.color.input);
		ret.strength.input = i.index_node(data.strength.input);

		return MaterialNodeSBTData{.emission = ret};
	}

};

inline std::shared_ptr<Node> make_node(const EmissionData& data){
	return std::make_shared<Emission>(data);
}



class Mix : public Node{
	MixData data;

public:
	Mix(const MixData& d):data(d){}

	int program()const{return 6;}
	int nprograms()const{return 3;}
	MaterialNodeSBTData sbtData(const NodeIndexingInfo& i){
		MixData ret = data;

		ret.shader1 = i.index_node(data.shader1);
		ret.shader2 = i.index_node(data.shader2);
		ret.factor.input = i.index_node(data.factor.input);

		return MaterialNodeSBTData{.mix = ret};
	}

};

inline std::shared_ptr<Node> make_node(const MixData& data){
	return std::make_shared<Mix>(data);
}




struct TextureCreateInfo{

	enum class Type{
		ImageTexture,
		Environment,
	};

	uint32_t image;
	cudaTextureDesc desc;
	Type type;

	TextureCreateInfo(
		uint32_t i,
		Type t = Type::ImageTexture,
		cudaTextureFilterMode interpolation = cudaFilterModeLinear,
		cudaTextureAddressMode extension = cudaAddressModeWrap)
	:image(i), type(t){
		desc.addressMode[0] = desc.addressMode[1] = extension;
		desc.filterMode = interpolation;
		desc.normalizedCoords = 1;
		desc.maxAnisotropy = 1;
		desc.maxMipmapLevelClamp = 99;
		desc.minMipmapLevelClamp = 0;
		desc.mipmapFilterMode = cudaFilterModePoint;
	}
};


class Texture_base : public Node{
	TextureCreateInfo info;
	cudaTextureObject_t cudaTexture {};

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

public:
	Texture_base(const TextureCreateInfo& i):info(i){};
	~Texture_base(){if(cudaTexture!=0)destroyCudaTexture();}

	int nprograms()const{return 1;}
	MaterialNodeSBTData sbtData(const NodeIndexingInfo& i){
		createCudaTexture(i.imageBuffers[info.image]);
		return MaterialNodeSBTData{.texture = cudaTexture};
	}
};

class ImageTexture : public Texture_base{
public:
	using Texture_base::Texture_base;
	~ImageTexture(){};
	int program()const{return 9;}
};


class EnvironmentTexture : public Texture_base{
public:
	using Texture_base::Texture_base;
	~EnvironmentTexture(){};
	int program()const{return 10;}
};


inline std::shared_ptr<Node> make_node(const TextureCreateInfo& info){
	if(info.type == TextureCreateInfo::Type::ImageTexture)
		return std::make_shared<ImageTexture>(info);
	else
		return std::make_shared<EnvironmentTexture>(info);
}


class Background : public Node{
	BackgroundData data;

public:
	Background(const BackgroundData& d):data(d){}

	int program()const{return 4;}
	int nprograms()const{return 1;}
	MaterialNodeSBTData sbtData(const NodeIndexingInfo& i){
		BackgroundData ret = data;

		ret.color.input = i.index_node(data.color.input);
		ret.strength.input = i.index_node(data.strength.input);

		return MaterialNodeSBTData{.background = ret};
	}

};

inline std::shared_ptr<Node> make_node(const BackgroundData& data){
	return std::make_shared<Background>(data);
}


}}
