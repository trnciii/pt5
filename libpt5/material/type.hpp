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
		const std::vector<std::string> names;
		NodeProgramManager(const std::vector<std::string>& n):id(-1), names(n){}
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

		int nprograms()const{
			int count = 0;
			for(const auto& node : nodes)
				count += node->nprograms();
			return count;
		}

		std::vector<int> offset_nodes()const{
			std::vector<int> ret(nodes.size());
			ret[0] = 0;
			for(int i=1; i<nodes.size(); i++)
				ret[i] = ret[i-1] + nodes[i-1]->nprograms();
			return ret;
		}
	};

}

using MaterialNode = material::Node;
using material::NodeIndexingInfo;
using material::Material;

}
