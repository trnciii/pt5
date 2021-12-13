#pragma once

#include <stdint.h>
#include "vector_math.h"


namespace pt5{

struct MTLData_Diffuse{
	float3 color = {0.6, 0.6, 0.6};
	uint32_t texture = 0;
};

struct MTLData_Emission{
	float3 color = {1,1,1};
	uint32_t texture = 0;
};

}