#pragma once

#include <vector>
#include "vector_math.h"
#include "CUDABuffer.hpp"

namespace pt5{

struct Camera{
	float3 position = {0,0,0};
	float3 toWorld[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
	float focalLength = 1;

	__device__ float3 view(float x, float y){
		float3 rayDir = make_float3(x, y, -focalLength);
		return normalize(make_float3(
			dot(toWorld[0], rayDir),
			dot(toWorld[1], rayDir),
			dot(toWorld[2], rayDir)));
	}

};


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

struct Material{
	float3 albedo = {0.6, 0.6, 0.6};
	float3 emission = {0, 0, 0};
};

struct TriangleMesh{
	std::vector<Vertex> vertices;
	std::vector<Face> indices;
	std::vector<float2> uv;
	std::vector<uint32_t> materialSlots;

	TriangleMesh(){}

	TriangleMesh(
		const std::vector<Vertex>& v,
		const std::vector<Face>& f,
		const std::vector<float2>& u,
		const std::vector<uint32_t>& m)
	:vertices(v), indices(f), uv(u), materialSlots(m){}
};


class Scene{
	std::vector<CUDABuffer> vertexBuffers;
	std::vector<CUDABuffer> indexBuffers;
	std::vector<CUDABuffer> uvBuffers;
	bool allocated = false;

public:

	float3 background = {0.4, 0.4, 0.4};
	std::vector<TriangleMesh> meshes;
	std::vector<Material> materials;


	~Scene(){
		if(allocated) free(0);
	}

	void upload(CUstream stream){
		assert(allocated == false);

		vertexBuffers.resize(meshes.size());
		indexBuffers.resize(meshes.size());
		uvBuffers.resize(meshes.size());

		for(int i=0; i<meshes.size(); i++){
			vertexBuffers[i].alloc_and_upload(meshes[i].vertices, stream);
			indexBuffers[i].alloc_and_upload(meshes[i].indices, stream);
			uvBuffers[i].alloc_and_upload(meshes[i].uv, stream);
		}

		allocated = true;
	}

	void free(CUstream stream){
		assert(allocated == true);
		for(auto& buffer : vertexBuffers) buffer.free(stream);
		for(auto& buffer : indexBuffers) buffer.free(stream);
		for(auto& buffer : uvBuffers) buffer.free(stream);

		allocated = false;
	}

	CUdeviceptr vertices_d_pointer(size_t i) const{
		assert(allocated);
		return vertexBuffers[i].d_pointer();
	}

	CUdeviceptr indices_d_pointer(size_t i) const{
		assert(allocated);
		return indexBuffers[i].d_pointer();
	}

	CUdeviceptr uv_d_pointer(size_t i) const{
		assert(allocated);
		return uvBuffers[i].d_pointer();
	}

};

}