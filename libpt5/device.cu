#include <optix_device.h>
#include <stdint.h>

#include "LaunchParams.h"
#include "vector_math.h"
#include "scene.hpp"

namespace pt5{

extern "C" __constant__ LaunchParams launchParams;


enum {
	SURFACE_RAY_TYPE=0,
	RAY_TYPE_COUNT
};


static __forceinline__ __device__
void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 ){
	const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__
void *unpackPointer(uint32_t i0, uint32_t i1){
	const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

template<typename T>
static __forceinline__ __device__
T* getPRD(){
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T*>(unpackPointer(u0, u1));
}


extern "C" __global__ void __closesthit__radiance(){
	float3& color = *(float3*)getPRD<float3>();
	color = make_float3(0.9, 0.8, 0.1);
}


extern "C" __global__ void __miss__radiance(){
	MissSBTData& sbtData = *(MissSBTData*)optixGetSbtDataPointer();
	float3* color = (float3*)getPRD<float3>();
	*color = sbtData.background;
}


extern "C" __global__ void __raygen__render(){
	RaygenSBTData& sbtData = *(RaygenSBTData*)optixGetSbtDataPointer();

	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;
	const uint2 frameSize = launchParams.image.size;
	const int pixel_index = iy*frameSize.x + ix;
	const Camera& camera = sbtData.camera;


	float3 color;
	uint32_t u0, u1;
	packPointer(&color, u0, u1);


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


	launchParams.image.pixels[pixel_index] = make_float4(color, 1);
}

} // pt5 namespace