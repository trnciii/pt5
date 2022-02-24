#pragma once

#include <iostream>
#include <stdint.h>
#include <vector>
#include "vector_math.h"
#include "CUDABuffer.hpp"

namespace pt5{

struct Vertex{
	float3 p;
	float3 n;
};

struct Face{
	uint3 vertices;
	uint3 uv;
	bool smooth = true;
	uint32_t material;
};

struct TriangleMesh{
	std::vector<Vertex> vertices;
	std::vector<Face> indices;
	std::vector<float2> uv;
	std::vector<uint32_t> materialSlots;

	CUDABuffer vertexBuffer;
	CUDABuffer indexBuffer;
	CUDABuffer uvBuffer;

	TriangleMesh(std::vector<Vertex> v, std::vector<Face> i, std::vector<float2> u, std::vector<uint32_t> m)
	:vertices(v), indices(i), uv(u), materialSlots(m){}

	~TriangleMesh(){
		if(vertexBuffer.allocated()) vertexBuffer.free(0);
		if(indexBuffer.allocated()) indexBuffer.free(0);
		if(uvBuffer.allocated()) uvBuffer.free(0);
	}

	void upload(CUstream stream=0){
		vertexBuffer.alloc_and_upload(vertices, stream);
		indexBuffer.alloc_and_upload(indices, stream);
		uvBuffer.alloc_and_upload(uv, stream);
	}

	void free(CUstream stream=0){
		vertexBuffer.free(stream);
		indexBuffer.free(stream);
		uvBuffer.free(stream);
	}
};

}