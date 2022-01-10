#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include "../sbt.hpp"


namespace pt5{
namespace material{

	struct NodeIndexingInfo{
		const int offset_material;
		const std::vector<int>& offset_nodes;
		const std::unordered_map<std::string, cudaArray_t>& imageBuffers;

		unsigned int index_node(const unsigned int i)const{
			if(i>0) return offset_material + offset_nodes[i];
			else return 0;
		}
	};


	struct NodeProgramManager{
		int id;
		const std::vector<const char*> names;
		NodeProgramManager(const std::vector<const char*>& n):id(-1), names(n){}
	};


	class Node{
	public:
		virtual int program()const=0;
		virtual int nprograms()const=0;
		virtual MaterialNodeSBTData sbtData(const NodeIndexingInfo&)=0;
	};


	class Material{
	public:
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
using material::NodeIndexingInfo;
using material::Material;

}
