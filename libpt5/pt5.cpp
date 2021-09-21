#include "pt5.hpp"

#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <iostream>
#include <stdexcept>


#define OPTIX_CHECK( call ){                                                                        \
	OptixResult res = call;                                                                         \
	if( res != OPTIX_SUCCESS ){                                                                     \
		fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
		exit( 2 );                                                                                  \
	}                                                                                               \
}


namespace pt5{

void nothing(){
	try{

		cudaFree(0);
		int numDevices;
		cudaGetDeviceCount(&numDevices);

		if(numDevices == 0)
			throw std::runtime_error("no device found");

		std::cout <<"found " <<numDevices <<" cuda device(s)" <<std::endl;
		OPTIX_CHECK( optixInit() );

		std::cout <<"optix initialized" <<std::endl;

	}catch(std::runtime_error& e){
		std::cout <<"error: " <<e.what() <<std::endl;
	}
}

int add(int a, int b){
	return a+b;
}

} // pt5 namespace