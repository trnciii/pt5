#include <optix_device.h>
#include <stdint.h>

#include "../vector_math.h"
#include "../LaunchParams.hpp"
#include "../sbt.hpp"
#include "intersection.cuh"
#include "util.cuh"
#include "math.cuh"


namespace pt5{

extern "C" __constant__ LaunchParams launchParams;

enum {
	SURFACE_RAY_TYPE=0,
	RAY_TYPE_COUNT
};


struct PaylaodData{
	float3 emission;
	float3 albedo;
	RNG rng;
	float pContinue;
	float3 ray_o;
	float3 ray_d;
};


extern "C" __global__ void __closesthit__radiance(){
	PaylaodData& payload = *(PaylaodData*)getPRD<PaylaodData>();
	const HitgroupSBTData& sbtData = *(HitgroupSBTData*)optixGetSbtDataPointer();

	const int primID = optixGetPrimitiveIndex();
	Intersection is = make_intersection(sbtData, primID);

	is.wi = optixDirectCall<float3, RNG&, const Intersection&>(sbtData.material+2, payload.rng, is);


	payload.emission = optixDirectCall<float3, const Intersection&>(sbtData.material+1, is);
	payload.albedo = optixDirectCall<float3, const Intersection&>(sbtData.material+0, is);
	payload.pContinue = max(payload.albedo.x, max(payload.albedo.y, payload.albedo.z));
	payload.ray_o = is.p + 0.001*is.ng;
	payload.ray_d = is.wi.x*is.tangent.x + is.wi.y*is.tangent.y + is.wi.z*is.n;
}


extern "C" __global__ void __miss__radiance(){
	MissSBTData& sbtData = *(MissSBTData*)optixGetSbtDataPointer();
	PaylaodData& payload = *(PaylaodData*)getPRD<PaylaodData>();

	Intersection is;
	is.n = optixGetWorldRayDirection();

	payload.emission = optixDirectCall<float3, const Intersection&>(sbtData, is);
	payload.albedo = make_float3(0);
	payload.pContinue = 0;
}


extern "C" __global__ void __raygen__render(){
	RaygenSBTData& sbtData = *(RaygenSBTData*)optixGetSbtDataPointer();

	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;
	const uint2 frameSize = launchParams.image.size;
	const int pixel_index = iy*frameSize.x + ix;

	PaylaodData payload;
	uint32_t u0, u1;
	packPointer(&payload, u0, u1);

	float3 accum = make_float3(0,0,0);
	payload.rng = RNG(pixel_index);

	for(int i=0; i<launchParams.spp; i++){

		const float x =  (2*(ix+payload.rng.uniform()) - frameSize.x)/frameSize.x;
		const float y = -(2*(iy+payload.rng.uniform()) - frameSize.y)/frameSize.x;

		payload.pContinue = 1;
		payload.ray_o = launchParams.camera.position;
		payload.ray_d = launchParams.camera.view(x,y);

		float3 throuput = make_float3(1);

		while(payload.rng.uniform() < payload.pContinue){

			throuput /= payload.pContinue;

			const float tmax = 1e20;
			const float tmin = 0;

			optixTrace(
				sbtData.traversable,
				payload.ray_o,
				payload.ray_d,
				tmin, tmax, 0,
				OptixVisibilityMask(255),
				OPTIX_RAY_FLAG_DISABLE_ANYHIT,
				SURFACE_RAY_TYPE,
				RAY_TYPE_COUNT,
				SURFACE_RAY_TYPE,
				u0, u1);

			accum += throuput * payload.emission;
			throuput *= payload.albedo;
		}
	}

	launchParams.image.pixels[pixel_index] = make_float4(accum/launchParams.spp, 1);
}

} // pt5 namespace