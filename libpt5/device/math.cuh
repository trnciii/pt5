#pragma once

#include <curand_kernel.h>

namespace pt5{
	
struct RNG{

	__device__ RNG(int i=0){
		curand_init(0, i, 0, &state);
	}

	__device__ float uniform(){
		return curand_uniform(&state);
	}

private:
	curandState state;
};


__device__ float3 sample_cosine_hemisphere(float u1, float u2){
	u2 *= 2*M_PI;
	float r = sqrt(u1);
	float z = sqrt(1-u1);
	return make_float3(r*cos(u2), r*sin(u2), z);
}


template <typename T>
__device__ __forceinline__ T barycentric(const T& x0, const T& x1, const T& x2, const float2& c){
	return (1-c.x-c.y)*x0 + c.x*x1 + c.y*x2;
}


}