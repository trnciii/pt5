#pragma once

#include "vector_math.h"
#include "mesh.hpp"
#include "material.h"
#include "camera.hpp"

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
	float2* uv;
	Material* material;
};

template <typename T>
struct Record{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

using RaygenRecord = Record<RaygenSBTData>;
using MissRecord = Record<MissSBTData>;
using HitgroupRecord = Record<HitgroupSBTData>;



struct LaunchParams{
	struct{
		float4* pixels;
		uint2 size;
	}image;

	uint32_t spp;

	Camera camera;
};

}