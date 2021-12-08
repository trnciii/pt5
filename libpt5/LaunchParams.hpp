#pragma once

#include "vector_math.h"
#include "camera.hpp"

namespace pt5{

struct LaunchParams{
	struct{
		float4* pixels;
		uint2 size;
	}image;

	uint32_t spp;

	Camera camera;
};

}