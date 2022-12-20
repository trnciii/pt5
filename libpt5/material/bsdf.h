#pragma once

#include "../vector_math.h"

namespace pt5{ namespace material{

__device__ float3 sample_cosine_hemisphere(float2 u){
	u.y *= 2*M_PI;
	float r = sqrt(u.x);
	float z = sqrt(1-u.x);
	return make_float3(r*cos(u.y), r*sin(u.y), z);
}

__device__ float fresnel(float ni, float eta){
	const float g2 = eta*eta - 1 + ni;
	if(g2 < 0) return 1;

	const float g = sqrt(g2);
	const float add = g + ni;
	const float sub = g - ni;

	const float frac = (ni*(add)-1)/(ni*sub+1);

	return 0.5 * ((sub*sub)/(add*add)) * (1 + frac*frac);
}

__device__ float beckmann_g1(float vn, float alpha){
	const float a = 1/(alpha * sqrt(1/(vn*vn) - 1));
	return (a<1.6)?
		(3.535*a + 2.181*a*a) / (1 + 2.276*a + 2.577*a*a)
		: 1;
}


__device__ float projected_angle(const float3& m, const float3& wi, const float3& z){
	float3 y = normalize(cross(z, wi));
	float3 x = cross(y, z);
	float3 mp = m - dot(y, m) * y;
	float s = dot(x, mp);
	float c = dot(z, mp);
	return atan2(s, c);
}

}}