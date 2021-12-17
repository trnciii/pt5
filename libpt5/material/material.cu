#include <optix_device.h>

#include "../vector_math.h"
#include "../kernel/intersection.cuh"
#include "data.h"

namespace pt5{

__device__ float3 sample_cosine_hemisphere(float2 u){
	u.y *= 2*M_PI;
	float r = sqrt(u.x);
	float z = sqrt(1-u.x);
	return make_float3(r*cos(u.y), r*sin(u.y), z);
}



extern "C" __device__ float3 __direct_callable__diffuse_albedo(const Intersection& is){
	BSDFData_Diffuse* material = (BSDFData_Diffuse*)is.materialData;
	return (material->color.texture>0)?
		make_float3(tex2D<float4>(material->color.texture, is.uv.x, is.uv.y))
		: material->color.default_value;
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
	BSDFData_Emission* material = (BSDFData_Emission*)is.materialData;
	return (material->color.texture>0)?
		make_float3(tex2D<float4>(material->color.texture, is.uv.x, is.uv.y))
		: material->color.default_value;
}

extern "C" __device__ float3 __direct_callable__emission_sample_direction(RNG& rng, const Intersection& is){
	return make_float3(0);
}


}