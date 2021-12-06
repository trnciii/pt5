#pragma once

#include "CUDABuffer.hpp"
#include "LaunchParams.h"


namespace pt5{

struct Scene;
class View;

class PathTracerState{
public:
	PathTracerState();
	~PathTracerState();

	void setScene(const Scene& scene);
	void removeScene();

	void render(const View& view, uint spp, const Camera& camera);

	bool running() const;

private:

	class SceneBuffer{
		std::vector<CUDABuffer> vertexBuffers;
		std::vector<CUDABuffer> indexBuffers;
		std::vector<CUDABuffer> uvBuffers;

	public:
		~SceneBuffer(){assert(allocated() == 0);}

		inline uint32_t allocated() const{
			assert((vertexBuffers.size() == indexBuffers.size())
				&& (vertexBuffers.size() == uvBuffers.size()));
			return vertexBuffers.size();
		}

		void upload(const Scene& scene, CUstream stream){
			vertexBuffers.resize(scene.meshes.size());
			indexBuffers.resize(scene.meshes.size());
			uvBuffers.resize(scene.meshes.size());

			for(int i=0; i<scene.meshes.size(); i++){
				vertexBuffers[i].alloc_and_upload(scene.meshes[i].vertices, stream);
				indexBuffers[i].alloc_and_upload(scene.meshes[i].indices, stream);
				uvBuffers[i].alloc_and_upload(scene.meshes[i].uv, stream);
			}

			cudaStreamSynchronize(stream);
		}

		void free(CUstream stream){
			for(CUDABuffer& buffer : vertexBuffers)buffer.free(stream);
			for(CUDABuffer& buffer : indexBuffers)buffer.free(stream);
			for(CUDABuffer& buffer : uvBuffers)buffer.free(stream);

			cudaStreamSynchronize(stream);
			vertexBuffers.clear();
			indexBuffers.clear();
			uvBuffers.clear();
		};

		inline CUdeviceptr vertices(int i) const{return vertexBuffers[i].d_pointer();}
		inline CUdeviceptr indices(int i) const{return indexBuffers[i].d_pointer();}
		inline CUdeviceptr uv(int i) const{return uvBuffers[i].d_pointer();}

	};

	void createContext();
	void createModule();
	void createProgramGroups();
	void createPipeline();

	void buildAccel(const std::vector<TriangleMesh>& meshes);
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


	OptixTraversableHandle asHandle;
	CUDABuffer asBuffer;

	SceneBuffer sceneBuffer;

	cudaEvent_t finishEvent;
};

}