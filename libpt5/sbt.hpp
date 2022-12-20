#pragma once

#include "optix.hpp"
#include "vector_math.h"
#include "mesh.hpp"
#include "material/data.h"

namespace pt5{


struct RaygenSBTData{
	OptixTraversableHandle traversable;
};


using MissSBTData = unsigned int;


struct HitgroupSBTData{
	Vertex* vertices;
	Face* faces;
	float2* uv;
	int material;
};

union MaterialNodeSBTData{
	material::DiffuseData diffuse;
	material::GlossyData glossy;
	material::MeasuredG1Data measuredG1;
	material::EmissionData emission;
	material::MixData mix;
	material::BackgroundData background;

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