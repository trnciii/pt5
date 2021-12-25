#pragma once

#include <memory>
#include <vector>
#include "data.h"
#include "../sbt.hpp"

namespace pt5{
namespace material{

	enum class Type{
		Diffuse,
		Emission,
		Mix,
	};



	struct Node{
		virtual size_t size()const=0;
		virtual void* ptr()const=0;
		virtual Type type()const=0;
		virtual int program()const=0;
		virtual int nprograms()const=0;
		virtual MaterialNodeSBTData sbtData(int offset_material, const std::vector<int>& offset_nodes)const=0;
	};


	struct Material{
		std::vector<std::shared_ptr<Node>> nodes;
		int offset_of_program(int n)const{
			int count = 0;
			for(int i=0; i<n; i++)
				count += nodes[i]->nprograms();
			return count;
		}
		int nprograms()const{return offset_of_program(nodes.size());}
	};




	struct Node_DiffuseBSDF : public Node{
		BSDFData_Diffuse data;

		Node_DiffuseBSDF():data(){}
		Node_DiffuseBSDF(const BSDFData_Diffuse& d):data(d){}

		size_t size() const{return sizeof(BSDFData_Diffuse);}
		void* ptr() const{return (void*)&data;}
		Type type()const {return Type::Diffuse;}
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

		Node_EmissionBSDF():data(){}
		Node_EmissionBSDF(const BSDFData_Emission& d):data(d){}

		size_t size() const{return sizeof(BSDFData_Emission);}
		void* ptr() const{return (void*)&data;}
		Type type()const{return Type::Emission;}
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

		Node_MixBSDF():data(){}
		Node_MixBSDF(const BSDFData_Mix& d):data(d){}

		size_t size()const{return sizeof(BSDFData_Mix);}
		void* ptr()const{return (void*)&data;}
		Type type()const{return Type::Mix;}
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

using MaterialType = material::Type;
using MaterialNode = material::Node;
using material::Material;
using material::make_node;


}
