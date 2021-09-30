#pragma once

#include <vector>
#include "vector_math.h"

namespace pt5{

struct Camera{
	float3 position = {0,0,0};
	float3 toWorld[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
	float focalLength = 1;
};

struct Material{
	float3 color;
};

struct TriangleMesh{
	std::vector<float3> vertex_coords;
	std::vector<float3> vertex_normals;

	std::vector<uint3> face_vertices;
	std::vector<uint32_t> face_material;

	std::vector<uint32_t> materials;
};


struct Scene{
	float3 background;
	Camera camera;
	std::vector<TriangleMesh> meshes;
	std::vector<Material> materials;
};

}