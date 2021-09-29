#include <optix_device.h>
#include <stdint.h>

#include "LaunchParams.h"
#include "vector_math.h"
#include "scene.hpp"
#include "deviceutil.cuh"

namespace pt5{

extern "C" __constant__ LaunchParams launchParams;


enum {
	SURFACE_RAY_TYPE=0,
	RAY_TYPE_COUNT
};


struct PaylaodData{
	float3 color;
};



extern "C" __global__ void __closesthit__radiance(){
	PaylaodData& prd = *(PaylaodData*)getPRD<PaylaodData>();
	const HitgroupSBTData& sbtData = *(HitgroupSBTData*)optixGetSbtDataPointer();

	const int primID = optixGetPrimitiveIndex();
	const Face& face = sbtData.indices[primID];

	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	const Vertex& v0 = sbtData.vertices[face.vertices.x];
	const Vertex& v1 = sbtData.vertices[face.vertices.y];
	const Vertex& v2 = sbtData.vertices[face.vertices.z];

	const Material& mtl = sbtData.material;

	const float3 p = (1-u-v)*v0.p + u*v1.p + v*v2.p;
	const float3 n = (1-u-v)*v0.n + u*v1.n + v*v2.n;

	prd.color = mtl.color;
}


extern "C" __global__ void __miss__radiance(){
	MissSBTData& sbtData = *(MissSBTData*)optixGetSbtDataPointer();
	PaylaodData& prd = *(PaylaodData*)getPRD<PaylaodData>();
	prd.color = sbtData.background;
}


extern "C" __global__ void __raygen__render(){
	RaygenSBTData& sbtData = *(RaygenSBTData*)optixGetSbtDataPointer();

	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;
	const uint2 frameSize = launchParams.image.size;
	const int pixel_index = iy*frameSize.x + ix;
	const Camera& camera = sbtData.camera;


	PaylaodData paload;
	uint32_t u0, u1;
	packPointer(&paload, u0, u1);


	const float x =  (2*(ix+0.5) - frameSize.x)/frameSize.x;
	const float y = -(2*(iy+0.5) - frameSize.y)/frameSize.x;
	float3 rayDir = make_float3(x, y, -camera.focalLength);
	rayDir = normalize(make_float3(
		dot(camera.toWorld[0], rayDir),
		dot(camera.toWorld[1], rayDir),
		dot(camera.toWorld[2], rayDir)));

	const float tmin = 0;
	const float tmax = 1e20;

	optixTrace(
		sbtData.traversable,
		camera.position, rayDir,
		tmin, tmax, 0,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_ANYHIT,
		SURFACE_RAY_TYPE,
		RAY_TYPE_COUNT,
		SURFACE_RAY_TYPE,
		u0, u1);


	launchParams.image.pixels[pixel_index] = make_float4(paload.color, 1);
}

} // pt5 namespace