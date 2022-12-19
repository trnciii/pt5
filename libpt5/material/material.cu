#include <optix_device.h>

#include "../vector_math.h"
#include "../kernel/intersection.cuh"
#include "../sbt.hpp"
#include "bsdf.h"
#include "data.h"

namespace pt5{ namespace material{

template <typename T>
__device__ T get_prop(const Prop<T>& t, const Intersection& is){
	return (t.input > 0)?
		optixDirectCall<T, const Intersection&>(t.input, is)
		: t.default_value;
}


extern "C" __device__ float3 __direct_callable__diffuse_albedo(const Intersection& is){
	const DiffuseData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->diffuse;
	return get_prop(material.color, is);
}

extern "C" __device__ float3 __direct_callable__diffuse_emission(const Intersection& is){
	return make_float3(0);
}

extern "C" __device__ float3 __direct_callable__diffuse_sample_direction(RNG& rng, const Intersection& is){
	return sample_cosine_hemisphere(rng.uniform2());
}



extern "C" __device__ float3 __direct_callable__beckmann_albedo(const Intersection& is){
	const GlossyData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->glossy;

	float alpha = get_prop(material.alpha, is);
	// const float F = fresnel(is.wi.z, 1.5);
	float G = beckmann_g1(is.wo.z, alpha) * beckmann_g1(is.wi.z, alpha);
	return G * get_prop(material.color, is);
}

extern "C" __device__ float3 __direct_callable__beckmann_emission(const Intersection& is){
	return make_float3(0);
}

extern "C" __device__ float3 __direct_callable__beckmann_sample_direction(RNG& rng, const Intersection& is){
	const GlossyData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->glossy;
	float alpha = get_prop(material.alpha, is);

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
	return get_prop(material.color, is)*get_prop(material.strength, is);
}

extern "C" __device__ float3 __direct_callable__emission_sample_direction(RNG& rng, const Intersection& is){
	return make_float3(0);
}




extern "C" __device__ float3 __direct_callable__mix_albedo(const Intersection& is){
	const MixData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->mix;
	float f = get_prop(material.factor, is);
	return (1-f)*optixDirectCall<float3, const Intersection&>(material.shader1, is)
		+ f*optixDirectCall<float3, const Intersection&>(material.shader2, is);
}

extern "C" __device__ float3 __direct_callable__mix_emission(const Intersection& is){
	const MixData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->mix;
	float f = get_prop(material.factor, is);
	return (1-f)*optixDirectCall<float3, const Intersection&>(material.shader1+1, is)
		+ f*optixDirectCall<float3, const Intersection&>(material.shader2+1, is);
}

extern "C" __device__ float3 __direct_callable__mix_sample_direction(RNG& rng, const Intersection& is){
	const MixData& material = ((MaterialNodeSBTData*)optixGetSbtDataPointer())->mix;
	if(rng.uniform() < get_prop(material.factor, is))
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