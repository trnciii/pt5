#pragma once

#include "vector_math.h"

namespace pt5{
namespace material{

	template <typename T>
	struct Prop{T default_value; long long texture = 0;};


	struct BSDFData_Diffuse{
		Prop<float3> color {{0.6, 0.6, 0.6}, -1};
	};


	struct BSDFData_Emission{
		Prop<float3> color {{1,1,1},-1};
		Prop<float> strength {1, -1};
	};


	struct BSDFData_Mix{
		int bsdf1 = -1;
		int bsdf2 = -1;
		Prop<float> factor {0.5, -1};
	};


}

using material::BSDFData_Diffuse;
using material::BSDFData_Emission;
using material::BSDFData_Mix;

}