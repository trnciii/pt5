#pragma once

#include <memory>
#include <vector>
#include "data.h"
#include "type.hpp"
#include "../sbt.hpp"


namespace pt5{
namespace material{

	inline void unwind_sbt_index(unsigned int& i, int offset_material, const std::vector<int>& offset_nodes){
		if(i>0)i = offset_material + offset_nodes[i];
	}

	struct Node_DiffuseBSDF : public Node{
		BSDFData_Diffuse data;

		Node_DiffuseBSDF(const BSDFData_Diffuse& d):data(d){}

		int program()const{return 0;}
		int nprograms()const{return 3;}
		MaterialNodeSBTData sbtData(int offset_material, const std::vector<int>& offset_nodes, const std::vector<cudaArray_t>& imageBuffers){
			BSDFData_Diffuse ret = data;

			unwind_sbt_index(ret.color.input, offset_material, offset_nodes);

			return MaterialNodeSBTData{.bsdf_diffuse = ret};
		}
	};

	inline std::shared_ptr<Node> make_node(const BSDFData_Diffuse& data){
		return std::make_shared<Node_DiffuseBSDF>(data);
	}



	struct Node_EmissionBSDF : public Node{
		BSDFData_Emission data;

		Node_EmissionBSDF(const BSDFData_Emission& d):data(d){}

		int program()const{return 3;}
		int nprograms()const{return 3;}
		MaterialNodeSBTData sbtData(int offset_material, const std::vector<int>& offset_nodes, const std::vector<cudaArray_t>& imageBuffers){
			BSDFData_Emission ret = data;

			unwind_sbt_index(ret.color.input, offset_material, offset_nodes);
			unwind_sbt_index(ret.strength.input, offset_material, offset_nodes);

			return MaterialNodeSBTData{.bsdf_emission = ret};
		}

	};

	inline std::shared_ptr<Node> make_node(const BSDFData_Emission& data){
		return std::make_shared<Node_EmissionBSDF>(data);
	}



	struct Node_MixBSDF : public Node{
		BSDFData_Mix data;

		Node_MixBSDF(const BSDFData_Mix& d):data(d){}

		int program()const{return 6;}
		int nprograms()const{return 3;}
		MaterialNodeSBTData sbtData(int offset_material, const std::vector<int>& offset_nodes, const std::vector<cudaArray_t>& imageBuffers){
			BSDFData_Mix ret = data;

			unwind_sbt_index(ret.bsdf1, offset_material, offset_nodes);
			unwind_sbt_index(ret.bsdf2, offset_material, offset_nodes);
			unwind_sbt_index(ret.factor.input, offset_material, offset_nodes);

			return MaterialNodeSBTData{.bsdf_mix = ret};
		}

	};

	inline std::shared_ptr<Node> make_node(const BSDFData_Mix& data){
		return std::make_shared<Node_MixBSDF>(data);
	}



	struct Node_ImageTexture : public Node{
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

		Node_ImageTexture(const CreateInfo& i):info(i){};

		~Node_ImageTexture(){if(cudaTexture!=0)destroyCudaTexture();}

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

	inline std::shared_ptr<Node> make_node(const Node_ImageTexture::CreateInfo& info){
		return std::make_shared<Node_ImageTexture>(info);
	}

}

using material::make_node;

using Texture = material::Node_ImageTexture::CreateInfo;

}
