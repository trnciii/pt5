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

	void setScene(const std::shared_ptr<Scene>);
	void removeScene();

	void render(const View& view, uint spp, const Camera& camera);
	inline void sync()const{CUDA_CHECK(cudaStreamSynchronize(stream));}

	inline void waitForRendering()const{
		CUDA_CHECK(cudaEventSynchronize(finishEvent));
	}

	inline void resetEvent(){
		CUDA_CHECK(cudaEventDestroy(finishEvent));
		finishEvent = nullptr;
	}

	inline bool launched()const{
		return finishEvent != nullptr;
	}

	inline bool running() const{
		return cudaEventQuery(finishEvent) == cudaErrorNotReady;
	}

	inline bool finished()const{
		return cudaEventQuery(finishEvent) == cudaSuccess;
	}


private:
	void createContext();
	void createModules();
	void createProgramGroups();
	void createPipeline();

	void buildAccel(const std::vector<std::shared_ptr<TriangleMesh>>& meshes);
	void buildSBT();

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

	std::shared_ptr<Scene> scene;

	CUDABuffer launchParamsBuffer;

	cudaEvent_t finishEvent = nullptr;
};

}