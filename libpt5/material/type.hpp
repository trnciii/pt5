#pragma once

#include <memory>
#include <vector>
#include "data.h"

namespace pt5{
namespace material{

	enum class Type{
		Diffuse,
		Emission,
	};


	struct Node{
		virtual size_t size()const=0;
		virtual void* ptr()const=0;
		virtual Type type()const=0;
		virtual int program()const=0;
		virtual int nprograms()const=0;
	};

	template <typename T>
	struct Node_t : Node{
		T data;

		Node_t():data(){}
		Node_t(const T& d):data(d){}

		size_t size() const{return sizeof(T);}
		void* ptr() const{return (void*)&data;}
		Type type()const;
		int program()const;
		int nprograms()const{return 3;}
	};


	template<>
	inline Type Node_t<BSDFData_Diffuse>::type()const{return Type::Diffuse;}

	template<>
	inline int Node_t<BSDFData_Diffuse>::program()const{return 0;}


	template<>
	inline Type Node_t<BSDFData_Emission>::type()const{return Type::Emission;}

	template<>
	inline int Node_t<BSDFData_Emission>::program()const{return 3;}


	template <typename T>
	inline std::shared_ptr<Node> make_node(const T& data){
		return std::make_shared<Node_t<T>>(Node_t(data));
	}


	struct Material{
		std::vector<std::shared_ptr<Node>> nodes;
		int nprograms()const{
			int count = 0;
			for(const std::shared_ptr<Node>& node : nodes)
				count += node->nprograms();
			return count;
		}
	};


}

using MaterialType = material::Type;
using MaterialNode = material::Node;
using material::make_node;
using Material = material::Material;


}
