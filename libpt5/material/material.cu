#include <optix_device.h>

#include "../vector_math.h"
#include "../kernel/intersection.cuh"
#include "../sbt.hpp"
#include "data.h"

namespace pt5{ namespace material{


__device__ float3 sample_cosine_hemisphere(float2 u){
	u.y *= 2*M_PI;
	float r = sqrt(u.x);
	float z = sqrt(1-u.x);
	return make_float3(r*cos(u.y), r*sin(u.y), z);
}



extern "C" __device__ float3 __direct_callable__diffuse_albedo(const Intersection& is){
	const DiffuseData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->diffuse;
	return (material.color.input>0)?
		optixDirectCall<float3, const Intersection&>(material.color.input, is)
		: material.color.default_value;
}

extern "C" __device__ float3 __direct_callable__diffuse_emission(const Intersection& is){
	return make_float3(0);
}

extern "C" __device__ float3 __direct_callable__diffuse_sample_direction(RNG& rng, const Intersection& is){
	return sample_cosine_hemisphere(rng.uniform2());
}




extern "C" __device__ float3 __direct_callable__emission_albedo(const Intersection& is){
	return make_float3(0);
}

extern "C" __device__ float3 __direct_callable__emission_emission(const Intersection& is){
	const EmissionData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->emission;

	float3 color = (material.color.input>0)?
		optixDirectCall<float3, const Intersection&>(material.color.input, is)
		: material.color.default_value;

	float strength = (material.strength.input>0)?
		optixDirectCall<float3, const Intersection&>(material.strength.input, is).x
		: material.strength.default_value;

	return color*strength;
}

extern "C" __device__ float3 __direct_callable__emission_sample_direction(RNG& rng, const Intersection& is){
	return make_float3(0);
}




extern "C" __device__ float3 __direct_callable__mix_albedo(const Intersection& is){
	const MixData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->mix;
	const float f = (material.factor.input>0)?
		optixDirectCall<float3, const Intersection&>(material.factor.input, is).x
		: material.factor.default_value;

	return (1-f)*optixDirectCall<float3, const Intersection&>(material.shader1, is)
		+ f*optixDirectCall<float3, const Intersection&>(material.shader2, is);
}

extern "C" __device__ float3 __direct_callable__mix_emission(const Intersection& is){
	const MixData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->mix;
	const float f = (material.factor.input>0)?
		optixDirectCall<float3, const Intersection&>(material.factor.input, is).x
		: material.factor.default_value;

	return (1-f)*optixDirectCall<float3, const Intersection&>(material.shader1+1, is)
		+ f*optixDirectCall<float3, const Intersection&>(material.shader2+1, is);
}

extern "C" __device__ float3 __direct_callable__mix_sample_direction(RNG& rng, const Intersection& is){
	const MixData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->mix;
	const float f = (material.factor.input>0)?
		optixDirectCall<float3, const Intersection&>(material.factor.input, is).x
		: material.factor.default_value;

	if(rng.uniform() < f)
		return optixDirectCall<float3, RNG&, const Intersection&>(material.shader1+2, rng, is);
	else
		return optixDirectCall<float3, RNG&, const Intersection&>(material.shader2+2, rng, is);
}



extern "C" __device__ float3 __direct_callable__image_texture(const Intersection& is){
	return make_float3(tex2D<float4>(((MaterialNodeSBTData*)optixGetSbtDataPointer())->texture, is.uv.x, is.uv.y));
}

}}