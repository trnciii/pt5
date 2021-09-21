#include <optix_device.h>

#include "LaunchParams.h"


namespace pt5{

extern "C" __constant__ LaunchParams launchParams;


extern "C" __global__ void __closesthit__radiance(){}
extern "C" __global__ void __anyhit__radiance(){}
extern "C" __global__ void __miss__radiance(){}

extern "C" __global__ void __raygen__render(){
	const int ix = optixGetLaunchIndex().x;
	const int iy = optixGetLaunchIndex().y;
	const int index = launchParams.image.width*iy + ix;

	launchParams.image.pixels[4*index  ] = float(ix%256)/256;
	launchParams.image.pixels[4*index+1] = float(iy%256)/256;
	launchParams.image.pixels[4*index+2] = 0.5;
	launchParams.image.pixels[4*index+3] = 1;
}

} // pt5 namespace