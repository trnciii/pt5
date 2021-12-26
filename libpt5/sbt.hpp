#pragma once

#include "optix.hpp"
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

union MaterialNodeSBTData{
	BSDFData_Diffuse bsdf_diffuse;
	BSDFData_Emission bsdf_emission;
	BSDFData_Mix bsdf_mix;

	cudaTextureObject_t texture;
};


template <typename T>
struct Record{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data {};
};

using RaygenRecord = Record<RaygenSBTData>;
using MissRecord = Record<MissSBTData>;
using HitgroupRecord = Record<HitgroupSBTData>;
using MaterialNodeRecord = Record<MaterialNodeSBTData>;

struct NullRecord{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};


}