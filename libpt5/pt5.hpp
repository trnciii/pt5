#pragma once

#include <cuda_gl_interop.h>
#include <vector>

#include "optix.hpp"
#include "CUDABuffer.hpp"
#include "LaunchParams.h"
#include "scene.hpp"


namespace pt5{


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