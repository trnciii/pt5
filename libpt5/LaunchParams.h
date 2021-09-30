#pragma once

#include "vector_math.h"
#include "scene.hpp"

namespace pt5{

struct RaygenSBTData{
	Camera camera;
	OptixTraversableHandle traversable;
};

struct MissSBTData{
	float3 background;
};

struct HitgroupSBTData{
	float3* vertex_coords;
	float3* vertex_normals;

	uint3* face_vertices;

	Material material;
};

struct LaunchParams{
	struct{
		float4* pixels;
		uint2 size;
	}image;
};

}