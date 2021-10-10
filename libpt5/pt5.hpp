#pragma once

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cassert>
#include <vector>

#include "LaunchParams.h"
#include "scene.hpp"


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


namespace pt5{

struct CUDABuffer {
	inline CUdeviceptr d_pointer() const
	{ return (CUdeviceptr)d_ptr; }

	//! re-size buffer to given number of bytes
	void resize(size_t size, CUstream stream)
	{
		if (d_ptr) free(stream);
		alloc(size, stream);
	}

	//! allocate to given number of bytes
	void alloc(size_t size, CUstream stream)
	{
		assert(d_ptr == nullptr);
		this->sizeInBytes = size;
		CUDA_CHECK(cudaMallocAsync( (void**)&d_ptr, sizeInBytes, stream));
	}

	//! free allocated memory
	void free(CUstream stream)
	{
		CUDA_CHECK(cudaFreeAsync(d_ptr, stream));
		d_ptr = nullptr;
		sizeInBytes = 0;
	}

	template<typename T>
	void alloc_and_upload(const std::vector<T> &vt, CUstream stream)
	{
		alloc(vt.size()*sizeof(T), stream);
		upload((const T*)vt.data(),vt.size(), stream);
	}

	template<typename T>
	void upload(const T *t, size_t count, CUstream stream)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes == count*sizeof(T));
		CUDA_CHECK(cudaMemcpyAsync(d_ptr, (void *)t, count*sizeof(T), cudaMemcpyHostToDevice, stream));
	}

	template<typename T>
	void download(T *t, size_t count, CUstream stream)
	{
		assert(d_ptr != nullptr);
		assert(sizeInBytes == count*sizeof(T));
		CUDA_CHECK(cudaMemcpyAsync((void *)t, d_ptr, count*sizeof(T), cudaMemcpyDeviceToHost, stream));
	}

	size_t sizeInBytes { 0 };
	void  *d_ptr { nullptr };
};



class View{
public:
	View(int w, int h);
	~View();

	void downloadImage();

	void initWindow();
	void showWindow();

	CUstream stream;

	int width;
	int height;
	std::vector<float> pixels;
	CUDABuffer pixelBuffer;

	cudaEvent_t* tracerFinishEvent;

	GLFWwindow* window;
};



class PathTracerState{
public:
	~PathTracerState();

	void init();
	void setScene(const Scene& scene);
	void initLaunchParams(View& view, const uint spp);

	void render();

	uint2 size(){return launchParams.image.size;}

private:
	void createContext();
	void createModule();
	void createProgramGroups();
	void createPipeline();

	void buildAccel(std::vector<TriangleMesh>);
	void buildSBT(const Scene& scene);


	OptixDeviceContext context;
	CUstream stream;

	OptixModule module;

	OptixPipeline pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions pipelineLinkOptions = {};


	OptixProgramGroup raygenProgramGroup;
	CUDABuffer raygenRecordBuffer;

	OptixProgramGroup missProgramGroup;
	CUDABuffer missRecordBuffer;

	OptixProgramGroup hitgroupProgramGroup;
	CUDABuffer hitgroupRecordsBuffer;

	OptixShaderBindingTable sbt = {};


	LaunchParams launchParams;
	CUDABuffer launchParamsBuffer;


	OptixTraversableHandle asHandle;
	CUDABuffer asBuffer;

	// buffer per geometry
	std::vector<CUDABuffer> vertexCoordsBuffers;
	std::vector<CUDABuffer> vertexNormalBuffers;
	std::vector<CUDABuffer> indexBuffers;

	cudaEvent_t finishEvent;
};

} // pt5 namespace