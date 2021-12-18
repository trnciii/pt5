#pragma once

#include <algorithm>

namespace pt5{

inline float linear_to_sRGB(float x){
	float y = (x > 0.0031308)?
		1.055 * (pow(x, (1.0 / 2.4))) - 0.055
	: 12.92 * x;
	return std::max(0.0f, std::min(1.0f, y));
}

}