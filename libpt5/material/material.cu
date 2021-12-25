#include <optix_device.h>

#include "../vector_math.h"
#include "../kernel/intersection.cuh"
#include "../sbt.hpp"
#include "data.h"

namespace pt5{

__device__ float3 sample_cosine_hemisphere(float2 u){
	u.y *= 2*M_PI;
	float r = sqrt(u.x);
	float z = sqrt(1-u.x);
	return make_float3(r*cos(u.y), r*sin(u.y), z);
}



extern "C" __device__ float3 __direct_callable__diffuse_albedo(const Intersection& is){
	const BSDFData_Diffuse& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->bsdf_diffuse;
	return (material.color.texture>0)?
		make_float3(tex2D<float4>(material.color.texture, is.uv.x, is.uv.y))
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
	const BSDFData_Emission& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->bsdf_emission;
	float3 color = (material.color.texture>0)?
		make_float3(tex2D<float4>(material.color.texture, is.uv.x, is.uv.y))
		: material.color.default_value;

	float strength = (material.strength.texture>0)?
		(tex2D<float4>(material.strength.texture, is.uv.x, is.uv.y)).x
		: material.strength.default_value;

	return color*strength;
}

extern "C" __device__ float3 __direct_callable__emission_sample_direction(RNG& rng, const Intersection& is){
	return make_float3(0);
}



extern "C" __device__ float3 __direct_callable__mix_albedo(const Intersection& is){
	const BSDFData_Mix& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->bsdf_mix;
	const float f = (material.factor.texture>0)?
		(tex2D<float4>(material.factor.texture, is.uv.x, is.uv.y)).x
		: material.factor.default_value;

	return (1-f)*optixDirectCall<float3, const Intersection&>(material.bsdf1, is)
		+ f*optixDirectCall<float3, const Intersection&>(material.bsdf2, is);
}

extern "C" __device__ float3 __direct_callable__mix_emission(const Intersection& is){
	const BSDFData_Mix& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->bsdf_mix;
	const float f = (material.factor.texture>0)?
		(tex2D<float4>(material.factor.texture, is.uv.x, is.uv.y)).x
		: material.factor.default_value;

	return (1-f)*optixDirectCall<float3, const Intersection&>(material.bsdf1+1, is)
		+ f*optixDirectCall<float3, const Intersection&>(material.bsdf2+1, is);
}

extern "C" __device__ float3 __direct_callable__mix_sample_direction(RNG& rng, const Intersection& is){
	const BSDFData_Mix& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->bsdf_mix;
	const float f = (material.factor.texture>0)?
		(tex2D<float4>(material.factor.texture, is.uv.x, is.uv.y)).x
		: material.factor.default_value;

	if(rng.uniform() < f)
		return optixDirectCall<float3, RNG&, const Intersection&>(material.bsdf1+2, rng, is);
	else
		return optixDirectCall<float3, RNG&, const Intersection&>(material.bsdf2+2, rng, is);
}


}