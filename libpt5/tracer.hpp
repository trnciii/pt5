#pragma once

#include "CUDABuffer.hpp"
#include "LaunchParams.hpp"
#include "mesh.hpp"
#include "scene.hpp"

namespace pt5{

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