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

	__device__ float2 uniform2(){
		return make_float2(curand_uniform(&state), curand_uniform(&state));
	}

	__device__ float3 uniform3(){
		return make_float3(curand_uniform(&state), curand_uniform(&state), curand_uniform(&state));
	}

private:
	curandState state;
};


template <typename T>
__device__ __forceinline__ T barycentric(const T& x0, const T& x1, const T& x2, const float2& c){
	return (1-c.x-c.y)*x0 + c.x*x1 + c.y*x2;
}

__device__ inline float2 equirectanglar(float3 v){
	v = normalize(v);
	if(v.z<=-1) return make_float2(0.5, 0);
	else if(1<=v.z) return make_float2(0.5, 1);
	else {
		float th = asinf(v.z);
		float cph = v.x/cosf(th);
		float ph = (0>v.y) ?acosf(cph) :-acosf(cph);
		return make_float2(ph*M_1_PI*0.5+0.5, th*M_1_PI+0.5);
	}
}

}