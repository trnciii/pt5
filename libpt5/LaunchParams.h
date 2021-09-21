#pragma once

namespace pt5{

struct LaunchParams{
	struct{
		float* pixels;
		int width;
		int height;
	}image;
};

}