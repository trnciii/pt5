#pragma once

#include <stdint.h>
#include <vector>
#include "vector_math.h"

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
};

}