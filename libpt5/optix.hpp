#pragma once

#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <stdexcept>
#include <sstream>
#include <cassert>


#define CUDA_CHECK(call)                            \
{                                                   \
  cudaError_t rc = call;                            \
  if (rc != cudaSuccess) {                          \
    std::stringstream txt;                          \
    cudaError_t err =  rc; /*cudaGetLastError();*/  \
    txt << "CUDA Error " << cudaGetErrorName(err)   \
        << " (" << cudaGetErrorString(err) << ")";  \
    throw std::runtime_error(txt.str());            \
  }                                                 \
}


#define OPTIX_CHECK( call )                                                                       \
{                                                                                                 \
  OptixResult res = call;                                                                         \
  if( res != OPTIX_SUCCESS )                                                                      \
    {                                                                                             \
      fprintf( stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res, __LINE__ ); \
      exit( 2 );                                                                                  \
    }                                                                                             \
}

#define CUDA_SYNC_CHECK()                                                                              \
{                                                                                                      \
	cudaDeviceSynchronize();                                                                             \
	cudaError_t error = cudaGetLastError();                                                              \
	if( error != cudaSuccess )                                                                           \
		{                                                                                                  \
			fprintf( stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString( error ) ); \
			exit( 2 );                                                                                       \
		}                                                                                                  \
}
