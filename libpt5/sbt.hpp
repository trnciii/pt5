#pragma once

#include "vector_math.h"
#include "mesh.hpp"
#include "material/data.h"

namespace pt5{


struct RaygenSBTData{
	OptixTraversableHandle traversable;
};

struct MissSBTData{
	float3 color;
	uint32_t texture;
	float strength;
};

struct HitgroupSBTData{
	Vertex* vertices;
	Face* faces;
	float2* uv;
	int material;
};

template <typename T>
struct Record{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

using RaygenRecord = Record<RaygenSBTData>;
using MissRecord = Record<MissSBTData>;
using HitgroupRecord = Record<HitgroupSBTData>;

struct NullRecord{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};


}