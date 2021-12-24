#pragma once

#include <memory>
#include <vector>

namespace pt5{
namespace material{

	enum class Type{
		Diffuse,
		Emission,
		Mix,
	};


	template <typename T>
	struct Prop{T default_value; long long texture = 0;};


	struct Node{
		virtual size_t size()const=0;
		virtual void* ptr()const=0;
		virtual Type type()const=0;
		virtual int program()const=0;
		virtual int nprograms()const=0;
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


}

using MaterialType = material::Type;
using MaterialNode = material::Node;
using Material = material::Material;


}
