#pragma once

#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <cuda_gl_interop.h>

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
	void alloc_and_upload(const T& t, CUstream stream){
		alloc(sizeof(T), stream);
		upload(&t, 1, stream);
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


class Scene{
	std::vector<CUDABuffer> vertexBuffers;
	std::vector<CUDABuffer> indexBuffers;
	std::vector<CUDABuffer> uvBuffers;
	bool allocated = false;

public:

	float3 background = {0.4, 0.4, 0.4};
	std::vector<TriangleMesh> meshes;
	std::vector<Material> materials;


	~Scene(){
		if(allocated) free(0);
	}

	void upload(CUstream stream){
		assert(allocated == false);

		vertexBuffers.resize(meshes.size());
		indexBuffers.resize(meshes.size());
		uvBuffers.resize(meshes.size());

		for(int i=0; i<meshes.size(); i++){
			vertexBuffers[i].alloc_and_upload(meshes[i].vertices, stream);
			indexBuffers[i].alloc_and_upload(meshes[i].indices, stream);
			uvBuffers[i].alloc_and_upload(meshes[i].uv, stream);
		}

		allocated = true;
	}

	void free(CUstream stream){
		assert(allocated == true);
		for(auto& buffer : vertexBuffers) buffer.free(stream);
		for(auto& buffer : indexBuffers) buffer.free(stream);
		for(auto& buffer : uvBuffers) buffer.free(stream);

		allocated = false;
	}

	CUdeviceptr vertices_d_pointer(size_t i) const{
		assert(allocated);
		return vertexBuffers[i].d_pointer();
	}

	CUdeviceptr indices_d_pointer(size_t i) const{
		assert(allocated);
		return indexBuffers[i].d_pointer();
	}

	CUdeviceptr uv_d_pointer(size_t i) const{
		assert(allocated);
		return uvBuffers[i].d_pointer();
	}

};


class View{
public:
	View(int w, int h);
	~View();

	uint2 size() const{return make_uint2(width, height);}
	float4* bufferPtr() const{return (float4*)pixelBuffer.d_pointer();}

	void createGLTexture();
	void updateGLTexture();
	void destroyGLTexture();
	bool hasGLTexture();
	GLuint GLTexture()const{return glTextureHandle;};

	void downloadImage();
	void clear(float4 c);

	std::vector<float> pixels;

private:
	CUstream stream;

	int width;
	int height;

	cudaGraphicsResource* cudaTextureResourceHandle = nullptr;
	GLuint glTextureHandle = GL_FALSE;
	CUDABuffer pixelBuffer;
};



class PathTracerState{
public:
	PathTracerState();
	~PathTracerState();

	void setScene(const Scene& scene);
	void removeScene();

	void render(const View& view, uint spp, Camera camera);

	bool running() const;

private:
	void createContext();
	void createModule();
	void createProgramGroups();
	void createPipeline();

	void buildAccel(const Scene&);
	void buildSBT(const Scene& scene);

	void destroyAccel();
	void destroySBT();


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

	cudaEvent_t finishEvent;
};

} // pt5 namespace