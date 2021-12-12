#include <optix_device.h>

#include "../vector_math.h"
#include "intersection.cuh"
#include "../material.h"

namespace pt5{

__device__ float3 sample_cosine_hemisphere(float u1, float u2){
	u2 *= 2*M_PI;
	float r = sqrt(u1);
	float z = sqrt(1-u1);
	return make_float3(r*cos(u2), r*sin(u2), z);
}

extern "C" __device__ float3 __direct_callable__albedo(const Intersection& is){
	Material* material = (Material*)is.materialData;
	return (material->texture>0)?
		make_float3(tex2D<float4>(material->texture, is.uv.x, is.uv.y))
		: material->albedo;
}

extern "C" __device__ float3 __direct_callable__emission(const Intersection& is){
	Material* material = (Material*)is.materialData;
	return material->emission;
}

extern "C" __device__ float3 __direct_callable__sample_direction(float u0, float u1, const Intersection& is){
	return sample_cosine_hemisphere(u0, u1);
}

}