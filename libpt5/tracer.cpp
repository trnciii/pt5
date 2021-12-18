#include "tracer.hpp"
#include <optix_function_table_definition.h>
#include <iostream>
#include <vector>

#include "optix.hpp"
#include "view.hpp"
#include "scene.hpp"
#include "camera.hpp"
#include "LaunchParams.hpp"
#include "sbt.hpp"


namespace pt5{


extern "C" char embedded_ptx_kernel[];
extern "C" char embedded_ptx_material[];


void context_log_callback(unsigned int level, const char *tag, const char *message, void *){
  fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
}


const std::vector<std::string> material_types{
	"diffuse",
	"emission",
};

const std::vector<std::string> material_methods{
	"albedo",
	"emission",
	"sample_direction"
};


void PathTracerState::createContext(){
	const int deviceID = 0;
	CUcontext cudaContext;

	CUDA_CHECK(cudaFree(0));
	OPTIX_CHECK(optixInit());

	CUDA_CHECK(cudaSetDevice(deviceID));
	CUDA_CHECK(cudaStreamCreate(&stream));

	CUresult res = cuCtxGetCurrent(&cudaContext);
	if(res != CUDA_SUCCESS)
		fprintf(stderr, "Error querying current context: %d\n", res);

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &context));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(context, context_log_callback, nullptr, 4));

	std::cout <<"created context" <<std::endl;
}


void PathTracerState::createModules(){
	OptixModuleCompileOptions moduleCompileOptions = {};
	moduleCompileOptions.maxRegisterCount = 50;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	pipelineCompileOptions = {};
	pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	pipelineCompileOptions.usesMotionBlur = false;
	pipelineCompileOptions.numPayloadValues = 2;
	pipelineCompileOptions.numAttributeValues = 2;
	pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";

	pipelineLinkOptions.maxTraceDepth = 2;


	const std::unordered_map<std::string, std::string> ptxCodes{
		{"kernel", embedded_ptx_kernel},
		{"material", embedded_ptx_material}
	};

	for(const auto& [key, code] : ptxCodes){
		OptixModule& module = modules[key];

		char log[2048];
		size_t sizeoflog = sizeof(log);
		OPTIX_CHECK(optixModuleCreateFromPTX(
			context,
			&moduleCompileOptions,
			&pipelineCompileOptions,
			code.c_str(), code.size(),
			log, &sizeoflog,
			&module
		));
		if(sizeoflog > 1)
			std::cout <<"log = " <<log <<std::endl;

		std::cout <<"created "<<key <<" module" <<std::endl;
	}
}



void PathTracerState::createProgramGroups(){
	OptixProgramGroupOptions options = {};

	{ // kernel
		std::vector<OptixProgramGroupDesc> descs;
		OptixModule& module = modules["kernel"];

		{
			OptixProgramGroupDesc& desc = descs.emplace_back(OptixProgramGroupDesc{});
			desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			desc.raygen.module = module;
			desc.raygen.entryFunctionName = "__raygen__render";
		}

		{
			OptixProgramGroupDesc& desc = descs.emplace_back(OptixProgramGroupDesc{});
			desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			desc.miss.module = module;
			desc.miss.entryFunctionName = "__miss__radiance";
		}

		{
			OptixProgramGroupDesc& desc = descs.emplace_back(OptixProgramGroupDesc{});
			desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			desc.hitgroup.moduleCH = module;
			desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		}

		kernelProgramGroups.resize(descs.size());
		char log[2048];
		size_t sizeoflog = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			context,
			descs.data(),
			descs.size(),
			&options,
			log, &sizeoflog,
			kernelProgramGroups.data()
			));

		if(sizeoflog>1)
			std::cout <<"log = " <<log <<std::endl;
	}


	{ // material
		std::vector<std::string> names;
		for(const std::string& type : material_types){
			for(const std::string& method : material_methods){
				names.push_back("__direct_callable__" + type + "_" + method);
			}
		}

		std::vector<OptixProgramGroupDesc> descs;
		for(const std::string& name : names){
			OptixProgramGroupDesc desc;
				desc.kind  = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
				desc.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
				desc.callables.moduleDC            = modules["material"];
				desc.callables.entryFunctionNameDC = name.c_str();
			descs.push_back(desc);
		}

		materialProgramGroups.resize(descs.size());

		char log[2048];
		size_t sizeoflog = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			context,
			descs.data(),
			descs.size(),
			&options,
			log, &sizeoflog,
			materialProgramGroups.data()
			));

		if(sizeoflog>1)
			std::cout <<"log = " <<log <<std::endl;

		std::cout <<"created material programs" <<std::endl;
	}
}


void PathTracerState::createPipeline(){
	std::vector<OptixProgramGroup> pgs;
	std::copy(kernelProgramGroups.begin(), kernelProgramGroups.end(), std::back_inserter(pgs));
	std::copy(materialProgramGroups.begin(), materialProgramGroups.end(), std::back_inserter(pgs));


	char log[2048];
	size_t sizeoflog = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(
		context,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		pgs.data(), pgs.size(),
		log, &sizeoflog,
		&pipeline
		));

	std::cout <<"created pipeline" <<std::endl;
}


void PathTracerState::buildSBT(const Scene& scene){

	// raygen
	{
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(kernelProgramGroups[0], &rec));
		rec.data.traversable = asHandle;

		raygenRecordBuffer.alloc_and_upload(rec, stream);

		sbt.raygenRecord = raygenRecordBuffer.d_pointer();
	}

	// miss
	{
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(kernelProgramGroups[1], &rec));
		rec.data.color = scene.background.color;
		rec.data.texture = scene.background.texture;
		rec.data.strength = scene.background.strength;

		missRecordBuffer.alloc_and_upload(rec, stream);

		sbt.missRecordBase = missRecordBuffer.d_pointer();
		sbt.missRecordStrideInBytes = sizeof(MissRecord);
		sbt.missRecordCount = 1;
	}

	// hitgroup
	{
		std::vector<HitgroupRecord> hitgroupRecords;
		for(int objectCount=0; objectCount<scene.meshes.size(); objectCount++){
			const TriangleMesh& mesh = scene.meshes[objectCount];
			int rayTypeCount = 1;

			std::vector<BSDF> materialData;
			if(mesh.materialSlots.size()==0)
				materialData.push_back(sceneBuffer.materialData_default());
			else{
				for(int i=0; i<mesh.materialSlots.size(); i++){
					materialData.push_back(sceneBuffer.materialData(mesh.materialSlots[i]));
				}
			}

			for(const BSDF& m : materialData){
				HitgroupRecord rec;

				HitgroupSBTData data = {
					(Vertex*)sceneBuffer.vertices(objectCount),
					(Face*)sceneBuffer.indices(objectCount),
					(float2*)sceneBuffer.uv(objectCount),
					m
				};

				OPTIX_CHECK(optixSbtRecordPackHeader(kernelProgramGroups[2], &rec));
				rec.data = data;

				hitgroupRecords.push_back(rec);
			}
		}

		hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords, stream);

		sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
		sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
	}

	{ // material
		std::vector<NullRecord> materialRecords;
		for(int i=0; i<material_methods.size()*material_types.size(); i++){
			NullRecord rec;
			OPTIX_CHECK(optixSbtRecordPackHeader(materialProgramGroups[i], &rec));
			materialRecords.push_back(rec);
		}
		materialRecordBuffer.alloc_and_upload(materialRecords, stream);

		sbt.callablesRecordBase = materialRecordBuffer.d_pointer();
		sbt.callablesRecordStrideInBytes = sizeof(NullRecord);
		sbt.callablesRecordCount = materialRecords.size();
	}

}

void PathTracerState::destroySBT(){
	raygenRecordBuffer.free(stream);
	missRecordBuffer.free(stream);
	hitgroupRecordsBuffer.free(stream);
	materialRecordBuffer.free(stream);
}


void PathTracerState::buildAccel(const std::vector<TriangleMesh>& meshes){
	std::vector<OptixBuildInput> triangleInput(meshes.size());
	std::vector<std::vector<uint32_t>> triangleInputFlags(meshes.size());
	std::vector<CUdeviceptr> d_vertices(meshes.size());

	for(int i=0; i<meshes.size(); i++){
		const TriangleMesh& mesh = meshes[i];
		const int materialSize = mesh.materialSlots.size()? mesh.materialSlots.size() : 1;

		d_vertices[i] = sceneBuffer.vertices(i) + offsetof(Vertex, p);
		triangleInputFlags[i] = std::vector<uint32_t>(materialSize, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);

		triangleInput[i] = {};
			triangleInput[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			triangleInput[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInput[i].triangleArray.numVertices = (int)mesh.vertices.size();
			triangleInput[i].triangleArray.vertexBuffers = &d_vertices[i];
			triangleInput[i].triangleArray.vertexStrideInBytes = sizeof(Vertex);

			triangleInput[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangleInput[i].triangleArray.numIndexTriplets = (int)mesh.indices.size();
			triangleInput[i].triangleArray.indexBuffer = sceneBuffer.indices(i) + offsetof(Face, vertices);
			triangleInput[i].triangleArray.indexStrideInBytes = sizeof(Face);

			triangleInput[i].triangleArray.flags = triangleInputFlags[i].data();
			triangleInput[i].triangleArray.numSbtRecords = materialSize;
			triangleInput[i].triangleArray.sbtIndexOffsetBuffer = sceneBuffer.indices(i) + offsetof(Face, material);
			triangleInput[i].triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
			triangleInput[i].triangleArray.sbtIndexOffsetStrideInBytes = sizeof(Face);
	}


	// blas
	OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accelOptions.motionOptions.numKeys = 1;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blasBufferSizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(
		context,
		&accelOptions,
		triangleInput.data(),
		(int)meshes.size(),
		&blasBufferSizes));


	// prepare compaction
	CUDABuffer compactedSizeBuffer;
	compactedSizeBuffer.alloc(sizeof(uint64_t), stream);

	OptixAccelEmitDesc emitDesc;
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.d_pointer();


	// build
	CUDABuffer tempBuffer;
	tempBuffer.alloc(blasBufferSizes.tempSizeInBytes, stream);

	CUDABuffer outputBuffer;
	outputBuffer.alloc(blasBufferSizes.outputSizeInBytes, stream);

	OPTIX_CHECK(optixAccelBuild(
		context,
		0,
		&accelOptions,
		triangleInput.data(),
		(int)meshes.size(),
		tempBuffer.d_pointer(), tempBuffer.sizeInBytes,
		outputBuffer.d_pointer(), outputBuffer.sizeInBytes,
		&asHandle,
		&emitDesc, 1
		));

	CUDA_SYNC_CHECK()


	// compaction
	uint64_t compactedSize;
	compactedSizeBuffer.download(&compactedSize, 1, stream);

	asBuffer.alloc(compactedSize, stream);
	OPTIX_CHECK(optixAccelCompact(
		context,
		0,
		asHandle,
		asBuffer.d_pointer(), asBuffer.sizeInBytes,
		&asHandle
		))

	CUDA_SYNC_CHECK()


	// cleanup
	outputBuffer.free(stream);
	tempBuffer.free(stream);
	compactedSizeBuffer.free(stream);
}

void PathTracerState::destroyAccel(){
	asBuffer.free(stream);
}


PathTracerState::PathTracerState(){
	createContext();
	createModules();
	createProgramGroups();
	createPipeline();

	std::cout <<"initialized PathTracerState" <<std::endl;
}

void PathTracerState::setScene(const Scene& scene){
	sceneBuffer.upload(scene, stream);
	buildAccel(scene.meshes);
	buildSBT(scene);
}

void PathTracerState::removeScene(){
	sceneBuffer.free(stream);
	destroyAccel();
	destroySBT();
}


PathTracerState::~PathTracerState(){
	removeScene();

	OPTIX_CHECK( optixPipelineDestroy( pipeline ) );

	for(int i=0; i<kernelProgramGroups.size(); i++)
		OPTIX_CHECK( optixProgramGroupDestroy(kernelProgramGroups[i]) );
	kernelProgramGroups.clear();

	for(int i=0; i<materialProgramGroups.size(); i++)
		OPTIX_CHECK( optixProgramGroupDestroy(materialProgramGroups[i]) );
	materialProgramGroups.clear();

	for(auto& [k, v] : modules)
		OPTIX_CHECK( optixModuleDestroy(v));
	modules.clear();

	OPTIX_CHECK( optixDeviceContextDestroy( context ) );

	std::cout <<"destroyed PathTracerState" <<std::endl;
}



void PathTracerState::render(const View& view, uint spp, const Camera& camera){
	LaunchParams params;
	params.image.size = view.size();
	params.image.pixels = view.bufferPtr();
	params.spp = spp;
	params.camera = camera;

	CUDABuffer buffer;
	buffer.alloc_and_upload(params, stream);


	cudaEventCreate(&finishEvent);

	OPTIX_CHECK(optixLaunch(
		pipeline,
		stream,
		buffer.d_pointer(),
		buffer.sizeInBytes,
		&sbt,
		params.image.size.x,
		params.image.size.y,
		1));

	buffer.free(stream);
	cudaEventRecord(finishEvent, stream);
}


bool PathTracerState::running() const{
	return cudaEventQuery(finishEvent) == cudaErrorNotReady;
}


} // pt5 namespace