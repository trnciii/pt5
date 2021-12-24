#pragma once

#include <memory>
#include <vector>

namespace pt5{
namespace material{

	enum class Type{
		Diffuse,
		Emission,
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
using Material = material::Material;


}
