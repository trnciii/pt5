#pragma once

#include "vector_math.h"

namespace pt5{ namespace material{

	template <typename T>
	struct Prop{T default_value; unsigned int input = 0;};


	struct DiffuseData{
		Prop<float3> color {{0.6, 0.6, 0.6}, 0};
	};

	struct GlossyData{
		Prop<float3> color {{0.6, 0.6, 0.6}, 0};
		Prop<float> alpha {0.2, 0};
	};


	struct EmissionData{
		Prop<float3> color {{1,1,1},0};
		Prop<float> strength {1, 0};
	};


	struct MixData{
		unsigned int shader1 = 0;
		unsigned int shader2 = 0;
		Prop<float> factor {0.5, 0};
	};


	struct BackgroundData{
		Prop<float3> color {{1,1,1},0};
		Prop<float> strength {1, 0};
	};

}}