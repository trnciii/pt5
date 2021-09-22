#include "pt5.hpp"
#include "util.hpp"

#include <vector>
#include <string>
#include <iostream>


int main(){

	const int width = 1200;
	const int height = 800;

	pt5::PathTracerState tracer;
	tracer.buildSBT();
	tracer.initLaunchParams(width, height);

	tracer.render();

	std::vector<float> pixels = tracer.pixels();
	pt5::writeImage("out_c++.png", width, height, pixels);
	std::cout <<"image saved" <<std::endl;

	return 0;
}