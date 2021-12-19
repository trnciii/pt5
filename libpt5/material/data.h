#pragma once

#include <stdint.h>
#include "optix.h"
#include "vector_math.h"


namespace pt5{
namespace material{

	struct BSDF{
		int albedo;
		int emission;
		int sample_direction;
	};


	template <typename T>
	struct Prop{T default_value; long long texture = 0;};


	struct BSDFData_Diffuse{
		Prop<float3> color {{0.6, 0.6, 0.6}, 0};
	};


	struct BSDFData_Emission{
		Prop<float3> color {{1,1,1},0};
		Prop<float> strength {1, 0};
	};

}

using BSDF = material::BSDF;
using BSDFData_Diffuse = material::BSDFData_Diffuse;
using BSDFData_Emission = material::BSDFData_Emission;

}