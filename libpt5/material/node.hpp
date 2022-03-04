#pragma once

#include <memory>
#include <vector>
#include <algorithm>
#include <string>

#include "data.h"
#include "type.hpp"
#include "../sbt.hpp"
#include "../image.hpp"


namespace pt5{ namespace material{

class DiffuseBSDF : public Node{
	DiffuseData data;

public:
	inline static NodeProgramManager pgManager = NodeProgramManager({
		"__direct_callable__diffuse_albedo",
		"__direct_callable__diffuse_emission",
		"__direct_callable__diffuse_sample_direction",
	});

	DiffuseBSDF(const DiffuseData& d):data(d){}

	int program()const{return pgManager.id;}
	int nprograms()const{return pgManager.names.size();}

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

	inline static NodeProgramManager pgManager = NodeProgramManager({
		"__direct_callable__emission_albedo",
		"__direct_callable__emission_emission",
		"__direct_callable__emission_sample_direction",
	});

	Emission(const EmissionData& d):data(d){}

	int program()const{return pgManager.id;}
	int nprograms()const{return pgManager.names.size();}

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

	inline static NodeProgramManager pgManager = NodeProgramManager({
		"__direct_callable__mix_albedo",
		"__direct_callable__mix_emission",
		"__direct_callable__mix_sample_direction",
	});

	int program()const{return pgManager.id;}
	int nprograms()const{return pgManager.names.size();}

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

	std::shared_ptr<Image> image;
	cudaTextureDesc desc;
	Type type;

	TextureCreateInfo(
		std::shared_ptr<Image> i,
		Type t = Type::ImageTexture,
		cudaTextureFilterMode interpolation = cudaFilterModeLinear,
		cudaTextureAddressMode extension = cudaAddressModeWrap)
	:image(i), type(t)
	{
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

	cudaTextureObject_t createCudaTexture(){
		assert(cudaTexture == 0);

		if(info.image->array == 0) info.image->upload();

		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = info.image->array;
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

	MaterialNodeSBTData sbtData(const NodeIndexingInfo& i){
		createCudaTexture();
		return MaterialNodeSBTData{.texture = cudaTexture};
	}
};


class ImageTexture : public Texture_base{
public:
	inline static NodeProgramManager pgManager = NodeProgramManager({
		"__direct_callable__image_texture",
	});

	using Texture_base::Texture_base;
	~ImageTexture(){};

	int program()const{return pgManager.id;}
	int nprograms()const{return pgManager.names.size();}
};


class EnvironmentTexture : public Texture_base{
public:
	inline static NodeProgramManager pgManager = NodeProgramManager({
		"__direct_callable__environment_texture",
	});

	using Texture_base::Texture_base;
	~EnvironmentTexture(){};

	int program()const{return pgManager.id;}
	int nprograms()const{return pgManager.names.size();}
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
	inline static NodeProgramManager pgManager = NodeProgramManager({
		"__direct_callable__emission_emission",
	});

	Background(const BackgroundData& d):data(d){}

	int program()const{return pgManager.id;}
	int nprograms()const{return pgManager.names.size();}

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



inline std::vector<std::string> nodeProgramNames(){
	const std::vector<std::reference_wrapper<material::NodeProgramManager>> nodePrograms{
		material::DiffuseBSDF::pgManager,
		material::Emission::pgManager,
		material::Mix::pgManager,
		material::ImageTexture::pgManager,
		material::EnvironmentTexture::pgManager,
		material::Background::pgManager,
	};

	std::vector<std::string> names;
	for(material::NodeProgramManager& node : nodePrograms){
		std::copy(node.names.begin(), node.names.end(), std::back_inserter(names));
	}

	return names;
}

inline void setNodeIndices(){
	const std::vector<std::reference_wrapper<material::NodeProgramManager>> nodePrograms{
		material::DiffuseBSDF::pgManager,
		material::Emission::pgManager,
		material::Mix::pgManager,
		material::ImageTexture::pgManager,
		material::EnvironmentTexture::pgManager,
		material::Background::pgManager,
	};

	int i=0;
	for(material::NodeProgramManager& node : nodePrograms){
		node.id = i;
		i += node.names.size();
	}
}



}}
