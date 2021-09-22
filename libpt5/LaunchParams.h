#pragma once

#include "vector_math.h"

namespace pt5{

struct LaunchParams{
	struct{
		float4* pixels;
		uint2 size;
	}image;
};

}