#include "pt5.hpp"
#include "util.hpp"

#include <vector>
#include <string>
#include <iostream>


int main(){

	const int width = 1200;
	const int height = 800;

	pt5::PathTracerState state;
	pt5::initPathTracer(state);
	pt5::buildSBT(state);
	pt5::initLaunchParams(state, width, height);


	OPTIX_CHECK(optixLaunch(
		state.pipeline,
		state.stream,
		state.launchParamsBuffer.d_pointer(),
		state.launchParamsBuffer.sizeInBytes,
		&state.sbt,
		width, height, 1));

	cudaDeviceSynchronize();
	cudaError_t e = cudaGetLastError();
	if(e != CUDA_SUCCESS){
		fprintf( stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString( e ) );
		exit( 2 );
	}

	std::cout <<"rendered" <<std::endl;


	std::vector<float>pixels = pt5::getPixels(state);
	pt5::writeImage("out.png", width, height, pixels);
	std::cout <<"image saved" <<std::endl;


	pt5::destroyPathTracer(state);

	return 0;
}