#pragma once

#include <vector>
#include "vector_math.h"

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
	uint32_t material;
};

struct Material{
	float3 albedo;
	float3 emission;
};

struct TriangleMesh{
	std::vector<Vertex> vertices;
	std::vector<Face> indices;
	std::vector<uint32_t> materialSlots;

	TriangleMesh(){}

	TriangleMesh(
		const std::vector<Vertex>& v,
		const std::vector<Face>& f,
		const std::vector<uint32_t>& m)
	:vertices(v), indices(f), materialSlots(m){}

	TriangleMesh(
		const std::vector<float3>& v,
		const std::vector<float3>& n,
		const std::vector<uint3>& f,
		const std::vector<uint32_t>& mIdx,
		const std::vector<uint32_t>& mSlt)
	:materialSlots(mSlt)
	{
		assert(v.size() == n.size());
		assert(f.size() == mIdx.size());

		vertices.resize(v.size());
		indices.resize(f.size());

		for(int i=0; i<v.size(); i++){
			vertices[i].p = v[i];
			vertices[i].n = n[i];
		}

		for(int i=0; i<f.size(); i++){
			indices[i].vertices = f[i];
			indices[i].material = mIdx[i];
		}
	}
};


struct Scene{
	float3 background;
	std::vector<TriangleMesh> meshes;
	std::vector<Material> materials;
};

}