#pragma once

#include "vector_math.h"

namespace pt5{
namespace material{

	template <typename T>
	struct Prop{T default_value; unsigned int input = 0;};


	struct BSDFData_Diffuse{
		Prop<float3> color {{0.6, 0.6, 0.6}, 0};
	};


	struct BSDFData_Emission{
		Prop<float3> color {{1,1,1},0};
		Prop<float> strength {1, 0};
	};


	struct BSDFData_Mix{
		unsigned int bsdf1 = 0;
		unsigned int bsdf2 = 0;
		Prop<float> factor {0.5, 0};
	};


}

using material::BSDFData_Diffuse;
using material::BSDFData_Emission;
using material::BSDFData_Mix;

}