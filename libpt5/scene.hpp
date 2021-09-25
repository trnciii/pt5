#pragma once

#include <vector>
#include "vector_math.h"

namespace pt5{

struct Camera{
	float3 position = {0,0,0};
	float3 toWorld[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
	float focalLength = 1;
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
	float3 color;
};

struct TriangleMesh{
	std::vector<Vertex> vertices;
	std::vector<Face> indices;
};


struct Scene{
	float3 background;
	Camera camera;
	std::vector<TriangleMesh> meshes;
	std::vector<Material> materials;
};

}