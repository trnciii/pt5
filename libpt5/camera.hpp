#pragma once

#include "vector_math.h"

namespace pt5{

struct Camera{
	float3 position = {0,0,0};
	float3 toWorld[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
	float focalLength = 1;

	__device__ float3 view(float x, float y){
		float3 rayDir = make_float3(x, y, -focalLength);
		return normalize(make_float3(
			dot(toWorld[0], rayDir),
			dot(toWorld[1], rayDir),
			dot(toWorld[2], rayDir)));
	}

};

}