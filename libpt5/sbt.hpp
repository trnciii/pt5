#pragma once

#include "vector_math.h"
#include "mesh.hpp"

namespace pt5{

struct Material;

struct RaygenSBTData{
	OptixTraversableHandle traversable;
};

struct MissSBTData{
	float3 background;
};

struct MaterialSBTData{
	CUdeviceptr data;
	int dc_albedo_id;
	int dc_emission_id;
	int dc_sample_direction_id;
};

struct HitgroupSBTData{
	Vertex* vertices;
	Face* faces;
	float2* uv;
	MaterialSBTData material;
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