#pragma once

#include <memory>
#include <vector>
#include "../sbt.hpp"


namespace pt5{
namespace material{

	struct Node{
		virtual int program()const=0;
		virtual int nprograms()const=0;
		virtual MaterialNodeSBTData sbtData(int, const std::vector<int>&, const std::vector<cudaArray_t>&)=0;
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

using MaterialNode = material::Node;
using material::Material;

}
