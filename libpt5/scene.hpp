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

}