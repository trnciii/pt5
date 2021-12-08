#pragma once

#include "vector_math.h"


namespace pt5{

struct Material{
	float3 albedo = {0.6, 0.6, 0.6};
	float3 emission = {0, 0, 0};
	uint32_t texture = 0;
};

}