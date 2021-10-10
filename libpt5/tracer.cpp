#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "pt5.hpp"

#include <optix_function_table_definition.h>
#include <chrono>

extern "C" char embedded_ptx_code[];


namespace pt5{


template <typename T>
struct Record{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

using RaygenRecord = Record<RaygenSBTData>;
using MissRecord = Record<MissSBTData>;
using HitgroupRecord = Record<HitgroupSBTData>;


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
		rec.data.camera = scene.camera;
		rec.data.traversable = asHandle;

		raygenRecordBuffer.alloc(sizeof(RaygenRecord), stream);
		raygenRecordBuffer.upload(&rec, 1, stream);

		sbt.raygenRecord = raygenRecordBuffer.d_pointer();
	}

	// miss
	{
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missProgramGroup, &rec));
		rec.data.background = scene.background;

		missRecordBuffer.alloc(sizeof(MissRecord), stream);
		missRecordBuffer.upload(&rec, 1, stream);

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

			for(int materialCount=0; materialCount<mesh.materials.size(); materialCount++){
				HitgroupRecord rec;

				HitgroupSBTData data = {
					(float3*)vertexCoordsBuffers[objectCount].d_pointer(),
					(float3*)vertexNormalBuffers[objectCount].d_pointer(),

					(uint3*)indexBuffers[objectCount].d_pointer(),

					scene.materials[mesh.materials[materialCount]]
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


void PathTracerState::buildAccel(std::vector<TriangleMesh> meshes){
	vertexCoordsBuffers.resize(meshes.size());
	vertexNormalBuffers.resize(meshes.size());

	indexBuffers.resize(meshes.size());

	std::vector<CUDABuffer> materialBuffer(meshes.size());

	std::vector<OptixBuildInput> triangleInput(meshes.size());
	std::vector<std::vector<uint32_t>> triangleInputFlags(meshes.size());
	std::vector<CUdeviceptr> d_vertices(meshes.size());

	for(int i=0; i<meshes.size(); i++){
		const TriangleMesh& mesh = meshes[i];

		vertexCoordsBuffers[i].alloc_and_upload(mesh.vertex_coords, stream);
		vertexNormalBuffers[i].alloc_and_upload(mesh.vertex_normals, stream);
		d_vertices[i] = vertexCoordsBuffers[i].d_pointer();

		indexBuffers[i].alloc_and_upload(mesh.face_vertices, stream);

		materialBuffer[i].alloc_and_upload(mesh.face_material, stream);

		triangleInputFlags[i] = std::vector<uint32_t>(mesh.materials.size(), OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);


		triangleInput[i] = {};
			triangleInput[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			triangleInput[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInput[i].triangleArray.numVertices = (int)mesh.vertex_coords.size();
			triangleInput[i].triangleArray.vertexBuffers = &d_vertices[i];
			triangleInput[i].triangleArray.vertexStrideInBytes = sizeof(float3);

			triangleInput[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangleInput[i].triangleArray.numIndexTriplets = (int)mesh.face_vertices.size();
			triangleInput[i].triangleArray.indexBuffer = indexBuffers[i].d_pointer();
			triangleInput[i].triangleArray.indexStrideInBytes = sizeof(uint3);

			triangleInput[i].triangleArray.flags = triangleInputFlags[i].data();
			triangleInput[i].triangleArray.numSbtRecords = mesh.materials.size();
			triangleInput[i].triangleArray.sbtIndexOffsetBuffer = materialBuffer[i].d_pointer();
			triangleInput[i].triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
			triangleInput[i].triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
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

	for(int i=0; i<materialBuffer.size(); i++){
		materialBuffer[i].free(stream);
	}
}


void PathTracerState::init(){
	createContext();
	createModule();
	createProgramGroups();
	createPipeline();

	std::cout <<"initialized PathTracerState" <<std::endl;
}

void PathTracerState::setScene(const Scene& scene){
	buildAccel(scene.meshes);
	buildSBT(scene);
}


PathTracerState::~PathTracerState(){
	OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
	OPTIX_CHECK( optixProgramGroupDestroy( raygenProgramGroup ) );
	OPTIX_CHECK( optixProgramGroupDestroy( missProgramGroup ) );
	OPTIX_CHECK( optixProgramGroupDestroy( hitgroupProgramGroup ) );
	OPTIX_CHECK( optixModuleDestroy( module ) );
	OPTIX_CHECK( optixDeviceContextDestroy( context ) );

	raygenRecordBuffer.free(stream);
	missRecordBuffer.free(stream);
	hitgroupRecordsBuffer.free(stream);
	launchParamsBuffer.free(stream);
	asBuffer.free(stream);

	for(int  i=0; i<vertexCoordsBuffers.size(); i++)
		vertexCoordsBuffers[i].free(stream);

	for(int  i=0; i<vertexNormalBuffers.size(); i++)
		vertexNormalBuffers[i].free(stream);

	for(int i=0; i<indexBuffers.size(); i++)
		indexBuffers[i].free(stream);

	std::cout <<"destroyed PathTracerState" <<std::endl;
}


void PathTracerState::initLaunchParams(View& view, const uint spp){
	launchParams.image.size = make_uint2(view.width, view.height);
	launchParams.image.pixels = (float4*)view.pixelBuffer.d_pointer();
	launchParams.spp = spp;

	view.tracerFinishEvent = &finishEvent;
}



void PathTracerState::render(){
	cudaEventCreate(&finishEvent);

	launchParamsBuffer.alloc(sizeof(launchParams), stream);
	launchParamsBuffer.upload(&launchParams, 1, stream);

	OPTIX_CHECK(optixLaunch(
		pipeline,
		stream,
		launchParamsBuffer.d_pointer(),
		launchParamsBuffer.sizeInBytes,
		&sbt,
		launchParams.image.size.x,
		launchParams.image.size.y,
		1));

	cudaEventRecord(finishEvent, stream);
}


} // pt5 namespace