#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "pt5.hpp"

#include <optix_function_table_definition.h>


extern "C" char embedded_ptx_code[];


namespace pt5{

/*! SBT record for a raygen program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
{
	__align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// just a dummy value - later examples will use more interesting
	// data here
	void *data;
};

/*! SBT record for a miss program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
{
	__align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// just a dummy value - later examples will use more interesting
	// data here
	void *data;
};

/*! SBT record for a hitgroup program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
{
	__align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// just a dummy value - later examples will use more interesting
	// data here
	void *data;
};


void context_log_callback(unsigned int level, const char *tag, const char *message, void *){
	fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
}



void PathTracerState::createContext(){
	CUcontext cudaContext;

	CUDA_CHECK(cudaFree(0));
	OPTIX_CHECK(optixInit());

	const int deviceID = 0;
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
		desc.hitgroup.moduleAH = module;
		desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

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


void PathTracerState::buildSBT(){
	// raygen
	{
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygenProgramGroup, &rec));
		rec.data = nullptr;

		raygenRecordBuffer.alloc(sizeof(RaygenRecord));
		raygenRecordBuffer.upload(&rec, 1);

		sbt.raygenRecord = raygenRecordBuffer.d_pointer();
	}

	// miss
	{
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(missProgramGroup, &rec));
		rec.data = nullptr;

		missRecordBuffer.alloc(sizeof(MissRecord));
		missRecordBuffer.upload(&rec, 1);

		sbt.missRecordBase = missRecordBuffer.d_pointer();
		sbt.missRecordStrideInBytes = sizeof(MissRecord);
		sbt.missRecordCount = 1;
	}

	// hitgroup
	{
		int materialSlotSize = 1;
		int rayTypeCount = 1;
		std::vector<HitgroupRecord> hitgroupRecords;

		for(int i=0; i<materialSlotSize; i++){
			HitgroupRecord rec;
			const int index = i*rayTypeCount + 0; // add ray type offset

			OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupProgramGroup, &rec));
			rec.data = nullptr;
			hitgroupRecords.push_back(rec);
		}

		hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);

		sbt.hitgroupRecordBase = hitgroupRecordsBuffer.d_pointer();
		sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
	}
}


PathTracerState::PathTracerState(){
	createContext();
	createModule();
	createProgramGroups();
	createPipeline();

	std::cout <<"initialized PathTracerState" <<std::endl;
}


PathTracerState::~PathTracerState(){
    OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( raygenProgramGroup ) );
    OPTIX_CHECK( optixProgramGroupDestroy( missProgramGroup ) );
    OPTIX_CHECK( optixProgramGroupDestroy( hitgroupProgramGroup ) );
    OPTIX_CHECK( optixModuleDestroy( module ) );
    OPTIX_CHECK( optixDeviceContextDestroy( context ) );

    raygenRecordBuffer.free();
	missRecordBuffer.free();
	hitgroupRecordsBuffer.free();
	pixelBuffer.free();
	launchParamsBuffer.free();

	std::cout <<"destroyed PathTracerState" <<std::endl;
}


void PathTracerState::initLaunchParams(const int w, const int h){
	pixelBuffer.alloc(4*w*h*sizeof(float));

	launchParams.image.width = w;
	launchParams.image.height = h;
	launchParams.image.pixels = (float*)pixelBuffer.d_pointer();

	launchParamsBuffer.alloc(sizeof(launchParams));
	launchParamsBuffer.upload(&launchParams, 1);
}


std::vector<float> PathTracerState::pixels(){
	uint32_t len = 4*launchParams.image.width*launchParams.image.height;
	std::vector<float> pixels(len);
	pixelBuffer.download(pixels.data(), len);
	return pixels;
}


void PathTracerState::render(){
	OPTIX_CHECK(optixLaunch(
		pipeline,
		stream,
		launchParamsBuffer.d_pointer(),
		launchParamsBuffer.sizeInBytes,
		&sbt,
		launchParams.image.width,
		launchParams.image.height,
		1));

	cudaDeviceSynchronize();
	cudaError_t e = cudaGetLastError();
	if(e != CUDA_SUCCESS){
		fprintf( stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString( e ) );
		exit( 2 );
	}

	std::cout <<"rendered" <<std::endl;
}

} // pt5 namespace