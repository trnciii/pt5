#pragma once

#include <memory>
#include <vector>
#include "data.h"
#include "type.hpp"
#include "../sbt.hpp"

namespace pt5{
namespace material{

	struct Node_DiffuseBSDF : public Node{
		BSDFData_Diffuse data;

		Node_DiffuseBSDF(const BSDFData_Diffuse& d):data(d){}

		int program()const{return 0;}
		int nprograms()const{return 3;}
		MaterialNodeSBTData sbtData(int offset_material, const std::vector<int>& offset_nodes)const{
			return MaterialNodeSBTData{.bsdf_diffuse = data};
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
		MaterialNodeSBTData sbtData(int offset_material, const std::vector<int>& offset_nodes)const{
			return MaterialNodeSBTData{.bsdf_emission = data};
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
		MaterialNodeSBTData sbtData(int offset_material, const std::vector<int>& offset_nodes)const{
			BSDFData_Mix ret = data;
			ret.bsdf1 = offset_material + offset_nodes[ret.bsdf1];
			ret.bsdf2 = offset_material + offset_nodes[ret.bsdf2];
			return MaterialNodeSBTData{.bsdf_mix = ret};
		}

	};

	inline std::shared_ptr<Node> make_node(const BSDFData_Mix& data){
		return std::make_shared<Node_MixBSDF>(data);
	}

}

using material::make_node;

}
