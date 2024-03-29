#pragma once

#include <optix_device.h>

#include "../vector_math.h"
#include "../mesh.hpp"
#include "../sbt.hpp"
#include "math.cuh"


namespace pt5{

struct Intersection{
	float3 p;
	struct { float3 x, y;} tangent;
	float3 n;
	float3 ng;
	float2 parametric;
	float2 uv;
	float3 wo;
	float3 wi;
};


__device__ Intersection make_intersection(const HitgroupSBTData& sbtData, int primID){
	Intersection is;

	const Face& face = sbtData.faces[primID];

	is.parametric = optixGetTriangleBarycentrics();

	const Vertex& v0 = sbtData.vertices[face.vertices.x];
	const Vertex& v1 = sbtData.vertices[face.vertices.y];
	const Vertex& v2 = sbtData.vertices[face.vertices.z];

	const float2& tx0 = sbtData.uv[face.uv.x];
	const float2& tx1 = sbtData.uv[face.uv.y];
	const float2& tx2 = sbtData.uv[face.uv.z];


	is.p = barycentric(v0.p, v1.p, v2.p, is.parametric);

	is.uv = barycentric(tx0, tx1, tx2, is.parametric);

	is.ng = normalize(cross(v1.p-v0.p, v2.p-v0.p));
	is.n = face.smooth? normalize(barycentric(v0.n, v1.n, v2.n, is.parametric)) : is.ng;

	is.n = faceforward(is.n, is.wo, is.ng);
	is.ng = faceforward(is.ng, is.wo, is.ng);

	is.tangent.y = normalize((fabs(is.n.x) > fabs(is.n.z))?
		make_float3(-is.n.y, is.n.x, 0)
		: make_float3(0, -is.n.z, is.n.y));
	is.tangent.x = cross(is.tangent.y, is.n);

	const float3 wo_global = -optixGetWorldRayDirection();
	is.wo = make_float3(
		dot(wo_global, is.tangent.x),
		dot(wo_global, is.tangent.y),
		dot(wo_global, is.n)
	);

	return is;
}

}