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

struct HitgroupSBTData{};

struct LaunchParams{
	struct{
		float4* pixels;
		uint2 size;
	}image;
};

}