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



void createContext(PathTracerState& state){
	CUcontext cudaContext;

	CUDA_CHECK(cudaFree(0));
	OPTIX_CHECK(optixInit());

	const int deviceID = 0;
	CUDA_CHECK(cudaSetDevice(deviceID));
	CUDA_CHECK(cudaStreamCreate(&state.stream));

	CUresult res = cuCtxGetCurrent(&cudaContext);
	if(res != CUDA_SUCCESS)
		fprintf(stderr, "Error querying current context: %d\n", res);

	OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &state.context));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(state.context, context_log_callback, nullptr, 4));

	std::cout <<"created context" <<std::endl;
}


void createModule(PathTracerState& state){
	OptixModuleCompileOptions moduleCompileOptions = {};
	moduleCompileOptions.maxRegisterCount = 50;
	moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	state.pipelineCompileOptions = {};
	state.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	state.pipelineCompileOptions.usesMotionBlur = false;
	state.pipelineCompileOptions.numPayloadValues = 2;
	state.pipelineCompileOptions.numAttributeValues = 2;
	state.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	state.pipelineCompileOptions.pipelineLaunchParamsVariableName = "launchParams";

	state.pipelineLinkOptions.maxTraceDepth = 2;

	const std::string ptxCode = embedded_ptx_code;

	char log[2048];
	size_t sizeoflog = sizeof(log);
	OPTIX_CHECK(optixModuleCreateFromPTX(
		state.context,
		&moduleCompileOptions,
		&state.pipelineCompileOptions,
		ptxCode.c_str(), ptxCode.size(),
		log, &sizeoflog,
		&state.module
		));
	if(sizeoflog > 1)
		std::cout <<"log = " <<log <<std::endl;

	std::cout <<"created module" <<std::endl;
}



void createProgramGroups(PathTracerState& state){
	OptixProgramGroupOptions options = {};

	// raygen
	{
		OptixProgramGroupDesc desc = {};
		desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		desc.raygen.module = state.module;
		desc.raygen.entryFunctionName = "__raygen__render";

		char log[2048];
		size_t sizeoflog = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			state.context,
			&desc,
			1,
			&options,
			log, &sizeoflog,
			&state.raygenProgramGroup
			));

		if(sizeoflog>1)
			std::cout <<"log = " <<log <<std::endl;

		std::cout <<"created raygen programs" <<std::endl;
	}

	// miss
	{
		OptixProgramGroupDesc desc = {};
		desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		desc.miss.module = state.module;
		desc.miss.entryFunctionName = "__miss__radiance";

		char log[2048];
		size_t sizeoflog = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			state.context,
			&desc,
			1,
			&options,
			log, &sizeoflog,
			&state.missProgramGroup
			));

		if(sizeoflog>1)
			std::cout <<"log = " <<log <<std::endl;

		std::cout <<"created miss programs" <<std::endl;
	}

	// hitgroup
	{
		OptixProgramGroupDesc desc = {};
		desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		desc.hitgroup.moduleCH = state.module;
		desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
		desc.hitgroup.moduleAH = state.module;
		desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

		char log[2048];
		size_t sizeoflog = sizeof(log);
		OPTIX_CHECK(optixProgramGroupCreate(
			state.context,
			&desc,
			1,
			&options,
			log, &sizeoflog,
			&state.hitgroupProgramGroup
			));

		if(sizeoflog>1)
			std::cout <<"log = " <<log <<std::endl;

		std::cout <<"created hitgroup programs" <<std::endl;
	}
}


void createPipeline(PathTracerState& state){
	OptixProgramGroup programgroups[] = {
		state.raygenProgramGroup,
		state.missProgramGroup,
		state.hitgroupProgramGroup
	};

	char log[2048];
	size_t sizeoflog = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(
		state.context,
		&state.pipelineCompileOptions,
		&state.pipelineLinkOptions,
		programgroups, sizeof(programgroups)/sizeof(OptixProgramGroup),
		log, &sizeoflog,
		&state.pipeline
		));

	std::cout <<"created pipeline" <<std::endl;
}

void buildSBT(PathTracerState& state){
	// raygen
	{
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(state.raygenProgramGroup, &rec));
		rec.data = nullptr;

		state.raygenRecordBuffer.alloc(sizeof(RaygenRecord));
		state.raygenRecordBuffer.upload(&rec, 1);

		state.sbt.raygenRecord = state.raygenRecordBuffer.d_pointer();
	}

	// miss
	{
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(state.missProgramGroup, &rec));
		rec.data = nullptr;

		state.missRecordBuffer.alloc(sizeof(MissRecord));
		state.missRecordBuffer.upload(&rec, 1);

		state.sbt.missRecordBase = state.missRecordBuffer.d_pointer();
		state.sbt.missRecordStrideInBytes = sizeof(MissRecord);
		state.sbt.missRecordCount = 1;
	}

	// hitgroup
	{
		int materialSlotSize = 1;
		int rayTypeCount = 1;
		std::vector<HitgroupRecord> hitgroupRecords;

		for(int i=0; i<materialSlotSize; i++){
			HitgroupRecord rec;
			const int index = i*rayTypeCount + 0; // add ray type offset

			OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroupProgramGroup, &rec));
			rec.data = nullptr;
			hitgroupRecords.push_back(rec);
		}

		state.hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);

		state.sbt.hitgroupRecordBase = state.hitgroupRecordsBuffer.d_pointer();
		state.sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
		state.sbt.hitgroupRecordCount = (int)hitgroupRecords.size();
	}
}


void initPathTracer(PathTracerState& state){
	createContext(state);
	createModule(state);
	createProgramGroups(state);
	createPipeline(state);
}

void destroyPathTracer(PathTracerState& state){
    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygenProgramGroup ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.missProgramGroup ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.hitgroupProgramGroup ) );
    OPTIX_CHECK( optixModuleDestroy( state.module ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );

    state.raygenRecordBuffer.free();
	state.missRecordBuffer.free();
	state.hitgroupRecordsBuffer.free();
	state.pixelBuffer.free();
	state.launchParamsBuffer.free();
}


void initLaunchParams(PathTracerState& state, const int w, const int h){
	state.pixelBuffer.alloc(4*w*h*sizeof(float));

	state.launchParams.image.width = w;
	state.launchParams.image.height = h;
	state.launchParams.image.pixels = (float*)state.pixelBuffer.d_pointer();

	state.launchParamsBuffer.alloc(sizeof(state.launchParams));
	state.launchParamsBuffer.upload(&state.launchParams, 1);
}


std::vector<float> getPixels(PathTracerState& state){
	uint32_t len = 4*state.launchParams.image.width*state.launchParams.image.height;
	std::vector<float> pixels(len);
	state.pixelBuffer.download(pixels.data(), len);
	return pixels;
}


void nothing(){
	try{

		cudaFree(0);
		int numDevices;
		cudaGetDeviceCount(&numDevices);

		if(numDevices == 0)
			throw std::runtime_error("no device found");

		std::cout <<"found " <<numDevices <<" cuda device(s)" <<std::endl;
		OPTIX_CHECK( optixInit() );

		std::cout <<"optix initialized" <<std::endl;

	}catch(std::runtime_error& e){
		std::cout <<"error: " <<e.what() <<std::endl;
	}
}

int add(int a, int b){
	return a+b;
}

} // pt5 namespace