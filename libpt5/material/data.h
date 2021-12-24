#pragma once

#include <stdint.h>
#include "optix.h"
#include "vector_math.h"
#include "type.hpp"

namespace pt5{
namespace material{

	struct BSDFData_Diffuse{
		Prop<float3> color {{0.6, 0.6, 0.6}, 0};
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
	};

	inline std::shared_ptr<Node> make_node(const BSDFData_Diffuse& data){
		return std::make_shared<Node_DiffuseBSDF>(data);
	}



	struct BSDFData_Emission{
		Prop<float3> color {{1,1,1},0};
		Prop<float> strength {1, 0};
	};

	struct Node_EmissionBSDF : public Node{
		BSDFData_Emission data;

		Node_EmissionBSDF():data(){}
		Node_EmissionBSDF(const BSDFData_Emission& d):data(d){}

		size_t size() const{return sizeof(BSDFData_Emission);}
		void* ptr() const{return (void*)&data;}
		Type type()const{return Type::Emission;}
		int program()const{return 3;}
		int nprograms()const{return 3;}
	};


	inline std::shared_ptr<Node> make_node(const BSDFData_Emission& data){
		return std::make_shared<Node_EmissionBSDF>(data);
	}


}

using material::BSDFData_Diffuse;
using material::BSDFData_Emission;

using material::make_node;

}