#include "tracer.hpp"
#include <optix_function_table_definition.h>
#include <iostream>
#include <vector>

#include "optix.hpp"
#include "view.hpp"
#include "scene.hpp"


extern "C" char embedded_ptx_code[];


namespace pt5{


void context_log_callback(unsigned int level, const char *tag, const char *message, void *){
  fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
}


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


void PathTracerState::createModule(){
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

	const std::string ptxCode = embedded_ptx_code;

	char log[2048];
	size_t sizeoflog = sizeof(log);
	OPTIX_CHECK(optixModuleCreateFromPTX(
		context,
		&moduleCompileOptions,
		&pipelineCompileOptions,
		ptxCode.c_str(), ptxCode.size(),
		log, &sizeoflog,
		&module
		));
	if(sizeoflog > 1)
		std::cout <<"log = " <<log <<std::endl;

	std::cout <<"created module" <<std::endl;
}



void PathTracerState::createProgramGroups(){
	OptixProgramGroupOptions options = {};

	// raygen
	{
		OptixProgramGroupDesc desc = {};
		desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		desc.raygen.module = module;
		desc.raygen.entryFunctionName = "__raygen__render";

		char log[2048];
		size_t sizeoflog = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			context,
			&desc,
			1,
			&options,
			log, &sizeoflog,
			&raygenProgramGroup
			));

		if(sizeoflog>1)
			std::cout <<"log = " <<log <<std::endl;

		std::cout <<"created raygen programs" <<std::endl;
	}

	// miss
	{
		OptixProgramGroupDesc desc = {};
		desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		desc.miss.module = module;
		desc.miss.entryFunctionName = "__miss__radiance";

		char log[2048];
		size_t sizeoflog = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			context,
			&desc,
			1,
			&options,
			log, &sizeoflog,
			&missProgramGroup
			));

		if(sizeoflog>1)
			std::cout <<"log = " <<log <<std::endl;

		std::cout <<"created miss programs" <<std::endl;
	}

	// hitgroup
	{
		OptixProgramGroupDesc desc = {};
		desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		desc.hitgroup.moduleCH = module;
		desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";

		char log[2048];
		size_t sizeoflog = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			context,
			&desc,
			1,
			&options,
			log, &sizeoflog,
			&hitgroupProgramGroup
			));

		if(sizeoflog>1)
			std::cout <<"log = " <<log <<std::endl;

		std::cout <<"created hitgroup programs" <<std::endl;
	}
}


void PathTracerState::createPipeline(){
	OptixProgramGroup programgroups[] = {
		raygenProgramGroup,
		missProgramGroup,
		hitgroupProgramGroup
	};

	char log[2048];
	size_t sizeoflog = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(
		context,
		&pipelineCompileOptions,
		&pipelineLinkOptions,
		programgroups, sizeof(programgroups)/sizeof(OptixProgramGroup),
		log, &sizeoflog,
		&pipeline
		));

	std::cout <<"created pipeline" <<std::endl;
}


void PathTracerState::buildSBT(const Scene& scene){

	// raygen
	{
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenProgramGroup, &rec));
		rec.data.traversable = asHandle;

		raygenRecordBuffer.alloc_and_upload(rec, stream);

		sbt.raygenRecord = raygenRecordBuffer.d_pointer();
	}

	// miss
	{
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missProgramGroup, &rec));
		rec.data.background = scene.background;

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

			std::vector<Material> materials;
			if(mesh.materialSlots.size()==0)
				materials.push_back(Material());
			else{
				for(int i=0; i<mesh.materialSlots.size(); i++){
					materials.push_back(scene.materials[mesh.materialSlots[i]]);
				}
			}

			for(int i=0; i<materials.size(); i++){
				HitgroupRecord rec;

				HitgroupSBTData data = {
					(Vertex*)scene.vertices_d_pointer(objectCount),
					(Face*)scene.indices_d_pointer(objectCount),
					(float2*)scene.uv_d_pointer(objectCount),
					materials[i]
				};

				OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupProgramGroup, &rec));
				rec.data = data;

				hitgroupRecords.push_back(rec);
			}
		}

		hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords, stream);

		sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
		sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
	}

}

void PathTracerState::destroySBT(){
	raygenRecordBuffer.free(stream);
	missRecordBuffer.free(stream);
	hitgroupRecordsBuffer.free(stream);
}


void PathTracerState::buildAccel(const Scene& scene){
	std::vector<OptixBuildInput> triangleInput(scene.meshes.size());
	std::vector<std::vector<uint32_t>> triangleInputFlags(scene.meshes.size());
	std::vector<CUdeviceptr> d_vertices(scene.meshes.size());

	for(int i=0; i<scene.meshes.size(); i++){
		const TriangleMesh& mesh = scene.meshes[i];
		const int materialSize = mesh.materialSlots.size()? mesh.materialSlots.size() : 1;

		d_vertices[i] = scene.vertices_d_pointer(i) + offsetof(Vertex, p);

		triangleInputFlags[i] = std::vector<uint32_t>(materialSize, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);

		triangleInput[i] = {};
			triangleInput[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			triangleInput[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInput[i].triangleArray.numVertices = (int)mesh.vertices.size();
			triangleInput[i].triangleArray.vertexBuffers = &d_vertices[i];
			triangleInput[i].triangleArray.vertexStrideInBytes = sizeof(Vertex);

			triangleInput[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangleInput[i].triangleArray.numIndexTriplets = (int)mesh.indices.size();
			triangleInput[i].triangleArray.indexBuffer = scene.indices_d_pointer(i) + offsetof(Face, vertices);
			triangleInput[i].triangleArray.indexStrideInBytes = sizeof(Face);

			triangleInput[i].triangleArray.flags = triangleInputFlags[i].data();
			triangleInput[i].triangleArray.numSbtRecords = materialSize;
			triangleInput[i].triangleArray.sbtIndexOffsetBuffer = scene.indices_d_pointer(i) + offsetof(Face, material);
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
		(int)scene.meshes.size(),
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
		(int)scene.meshes.size(),
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
	createModule();
	createProgramGroups();
	createPipeline();

	std::cout <<"initialized PathTracerState" <<std::endl;
}

void PathTracerState::setScene(const Scene& scene){
	buildAccel(scene);
	buildSBT(scene);
}

void PathTracerState::removeScene(){
	destroyAccel();
	destroySBT();
}


PathTracerState::~PathTracerState(){
	removeScene();

	OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
	OPTIX_CHECK( optixProgramGroupDestroy( raygenProgramGroup ) );
	OPTIX_CHECK( optixProgramGroupDestroy( missProgramGroup ) );
	OPTIX_CHECK( optixProgramGroupDestroy( hitgroupProgramGroup ) );
	OPTIX_CHECK( optixModuleDestroy( module ) );
	OPTIX_CHECK( optixDeviceContextDestroy( context ) );

	std::cout <<"destroyed PathTracerState" <<std::endl;
}



void PathTracerState::render(const View& view, uint spp, Camera camera){
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