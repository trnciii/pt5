#include <optix_device.h>
#include <stdint.h>

#include "../vector_math.h"
#include "../LaunchParams.hpp"
#include "../material.h"
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
	const Face& face = sbtData.faces[primID];

	const float3 wi = -optixGetWorldRayDirection();

	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;

	const Vertex& v0 = sbtData.vertices[face.vertices.x];
	const Vertex& v1 = sbtData.vertices[face.vertices.y];
	const Vertex& v2 = sbtData.vertices[face.vertices.z];

	const float2& tx0 = sbtData.uv[face.uv.x];
	const float2& tx1 = sbtData.uv[face.uv.y];
	const float2& tx2 = sbtData.uv[face.uv.z];

	const Material& mtl = sbtData.material;

	const float3 p = (1-u-v)*v0.p + u*v1.p + v*v2.p;
	const float2 txcoord = (1-u-v)*tx0 + u*tx1 + v*tx2;
	float3 ng = normalize(cross(v1.p-v0.p, v2.p-v0.p));
	float3 n = face.smooth? normalize((1-u-v)*v0.n + u*v1.n + v*v2.n) : ng;

	n = faceforward(n, wi, ng);
	ng = faceforward(ng, wi, ng);

	float3 tangent;
	float3 binromal;
	if(fabs(n.x) > fabs(n.z)){
		binromal = make_float3(-n.y, n.x, 0);
	}
	else{
		binromal = make_float3(0, -n.z, n.y);
	}

	binromal = normalize(binromal);
	tangent = cross(binromal, n);

	float3 ray_d = sample_cosine_hemisphere(payload.rng.uniform(), payload.rng.uniform());


	float3 color;
	if(mtl.texture>0){
		color = make_float3(tex2D<float4>(mtl.texture, txcoord.x, txcoord.y));
	}
	else{
		color = mtl.albedo;
	}


	payload.pContinue = max(mtl.albedo.x, max(mtl.albedo.y, mtl.albedo.z));
	payload.emission = mtl.emission;
	payload.albedo = color;
	payload.ray_o = p + 0.001*ng;
	payload.ray_d = ray_d.x*tangent + ray_d.y*binromal + ray_d.z*n;
}


extern "C" __global__ void __miss__radiance(){
	MissSBTData& sbtData = *(MissSBTData*)optixGetSbtDataPointer();
	PaylaodData& payload = *(PaylaodData*)getPRD<PaylaodData>();
	payload.emission = sbtData.background;
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