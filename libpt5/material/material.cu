#include <optix_device.h>

#include "../vector_math.h"
#include "../kernel/intersection.cuh"
#include "data.h"

namespace pt5{

__device__ float3 sample_cosine_hemisphere(float u1, float u2){
	u2 *= 2*M_PI;
	float r = sqrt(u1);
	float z = sqrt(1-u1);
	return make_float3(r*cos(u2), r*sin(u2), z);
}



extern "C" __device__ float3 __direct_callable__diffuse_albedo(const Intersection& is){
	MTLData_Diffuse* material = (MTLData_Diffuse*)is.materialData;
	return (material->texture>0)?
		make_float3(tex2D<float4>(material->texture, is.uv.x, is.uv.y))
		: material->color;
}

extern "C" __device__ float3 __direct_callable__diffuse_emission(const Intersection& is){
	return make_float3(0);
}

extern "C" __device__ float3 __direct_callable__diffuse_sample_direction(float u0, float u1, const Intersection& is){
	return sample_cosine_hemisphere(u0, u1);
}




extern "C" __device__ float3 __direct_callable__emission_albedo(const Intersection& is){
	return make_float3(0);
}

extern "C" __device__ float3 __direct_callable__emission_emission(const Intersection& is){
	MTLData_Emission* material = (MTLData_Emission*)is.materialData;
	return (material->texture>0)?
		make_float3(tex2D<float4>(material->texture, is.uv.x, is.uv.y))
		: material->color;
}

extern "C" __device__ float3 __direct_callable__emission_sample_direction(float u0, float u1, const Intersection& is){
	return make_float3(0);
}


}