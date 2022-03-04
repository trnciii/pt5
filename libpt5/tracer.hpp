#pragma once

#include <unordered_map>
#include <memory>
#include <vector>

#include "CUDABuffer.hpp"
#include "mesh.hpp"
#include "material/type.hpp"

namespace pt5{

struct Camera;
class View;

struct Scene{
	Material background;
	std::vector<std::shared_ptr<TriangleMesh>> meshes;
	std::vector<Material> materials;
};

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
	void createModules();
	void createProgramGroups();
	void createPipeline();

	void buildAccel(const std::vector<std::shared_ptr<TriangleMesh>>& meshes);
	void buildSBT(const Scene& scene);

	void destroyAccel();
	void destroySBT();


	OptixDeviceContext context;
	CUstream stream;

	std::unordered_map<std::string, OptixModule> modules;

	OptixPipeline pipeline;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	OptixPipelineLinkOptions pipelineLinkOptions = {};


	std::vector<OptixProgramGroup> kernelProgramGroups;
	CUDABuffer raygenRecordBuffer;
	CUDABuffer missRecordBuffer;
	CUDABuffer hitgroupRecordsBuffer;

	std::vector<OptixProgramGroup> materialProgramGroups;
	CUDABuffer materialRecordBuffer;

	OptixShaderBindingTable sbt = {};


	OptixTraversableHandle asHandle;
	CUDABuffer asBuffer;

	CUDABuffer launchParamsBuffer;

	cudaEvent_t finishEvent;
};

}