#pragma once

#include "vector_math.h"
#include "scene.hpp"

namespace pt5{

struct RaygenSBTData{
	OptixTraversableHandle traversable;
};

struct MissSBTData{
	float3 background;
};

struct HitgroupSBTData{
	Vertex* vertices;
	Face* faces;
	Material material;
};

struct LaunchParams{
	struct{
		float4* pixels;
		uint2 size;
	}image;

	uint32_t spp;

	Camera camera;
};

}