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



extern "C" __device__ float3 __direct_callable__beckmann_albedo(const Intersection& is, float3 wi){
	const GlossyData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->glossy;

	const float alpha = (material.alpha.input>0)?
		optixDirectCall<float3, const Intersection&>(material.alpha.input, is).x
		: material.alpha.default_value;

	const float F = fresnel(is.wi.z, 1.5);
	const float G = beckmann_g1(is.wo.z, alpha) * beckmann_g1(is.wi.z, alpha);

	return F*G* (material.color.input>0)?
		optixDirectCall<float3, const Intersection&>(material.color.input, is)
		: material.color.default_value;
}

extern "C" __device__ float3 __direct_callable__beckmann_emission(const Intersection& is){
	return make_float3(0);
}

extern "C" __device__ float3 __direct_callable__beckmann_sample_direction(RNG& rng, const Intersection& is){
	const GlossyData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->glossy;
	const float alpha = (material.alpha.input>0)?
		optixDirectCall<float3, const Intersection&>(material.alpha.input, is).x
		: material.alpha.default_value;

	const auto [u1, u2] = rng.uniform2();
	const float phi = 2*M_PI*u2;
	const float y2 = - alpha * alpha * log(1-u1);
	const float r = sqrt(y2) / sqrt(1 + y2);

	const auto m = make_float3(r*cos(phi), r*sin(phi), 1/sqrt(1 + y2));

	return -is.wo + 2*dot(is.wo, m)*m;
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

extern "C" __device__ float3 __direct_callable__environment_texture(const Intersection& is){
	float2 co = equirectanglar(is.n);
	return make_float3(tex2D<float4>(((MaterialNodeSBTData*)optixGetSbtDataPointer())->texture, co.x, co.y));
}

}}