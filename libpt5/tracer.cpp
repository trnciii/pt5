#include <optix_function_table_definition.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include "tracer.hpp"
#include "optix.hpp"
#include "view.hpp"
#include "camera.hpp"
#include "LaunchParams.hpp"
#include "sbt.hpp"
#include "material/node.hpp"


namespace pt5{


extern "C" char embedded_ptx_kernel[];
extern "C" char embedded_ptx_material[];


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
		std::vector<std::string> names = material::nodeProgramNames();

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


void PathTracerState::buildSBT(){

	std::vector<int> offset_material;
	offset_material.resize(scene->materials.size()+1);
	offset_material[0] = 0;

	for(int i=0; i<scene->materials.size(); i++)
		offset_material[i+1] = offset_material[i] + scene->materials[i].nprograms();


	std::vector<std::vector<int>> offset_nodes(scene->materials.size());
	for(int m=0; m<scene->materials.size(); m++){
		const Material& material = scene->materials[m];
		std::vector<int>& offset = offset_nodes[m];

		offset.resize(material.nodes.size());
		for(int n=1; n<material.nodes.size(); n++)
			offset[n] = offset[n-1] + material.nodes[n-1]->nprograms();
	}

	// offset of all materials and default diffuse (has 3 programs)
	int offset_backgroud = offset_material[scene->materials.size()] + 3;
	std::vector<int> offset_backgroud_nodes(scene->background.nodes.size());
	for(int n=1; n<scene->background.nodes.size(); n++)
		offset_backgroud_nodes[n] = offset_backgroud_nodes[n-1] + scene->background.nodes[n-1]->nprograms();



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
		rec.data = offset_backgroud;

		missRecordBuffer.alloc_and_upload(rec, stream);

		sbt.missRecordBase = missRecordBuffer.d_pointer();
		sbt.missRecordStrideInBytes = sizeof(MissRecord);
		sbt.missRecordCount = 1;
	}


	// hitgroup
	{
		std::vector<HitgroupRecord> hitgroupRecords;
		for(int objectCount=0; objectCount<scene->meshes.size(); objectCount++){
			const TriangleMesh& mesh = *scene->meshes[objectCount];
			int rayTypeCount = 1;

			std::vector<uint32_t> materialIndices = mesh.materialSlots;
			if(materialIndices.size() == 0) materialIndices.push_back(scene->materials.size());

			for(const int i : materialIndices){
				HitgroupRecord rec;

				HitgroupSBTData data = {
					(Vertex*)mesh.vertexBuffer.d_pointer(),
					(Face*)mesh.indexBuffer.d_pointer(),
					(float2*)mesh.uvBuffer.d_pointer(),
					offset_material[i]
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
		std::vector<MaterialNodeRecord> materialRecords;
		for(int m=0; m<scene->materials.size(); m++){
			const Material& material = scene->materials[m];
			for(int n=0; n<material.nodes.size(); n++){
				const std::shared_ptr<MaterialNode>& node = material.nodes[n];
				for(int pg = 0; pg < node->nprograms(); pg++){
 					MaterialNodeRecord rec;
					OPTIX_CHECK(optixSbtRecordPackHeader(materialProgramGroups[node->program() + pg], &rec));
					rec.data = node->sbtData(NodeIndexingInfo{offset_material[m], offset_nodes[m]});
					materialRecords.push_back(rec);
				}
			}
		}

		{
			for(int i=0; i<3; i++){
				MaterialNodeRecord rec;
				OPTIX_CHECK(optixSbtRecordPackHeader(materialProgramGroups[i], &rec));
				rec.data = MaterialNodeSBTData{.diffuse = material::DiffuseData()};
				materialRecords.push_back(rec);
			}
		}

		{
			for(int n=0; n<scene->background.nodes.size(); n++){
				const std::shared_ptr<MaterialNode>& node = scene->background.nodes[n];
				for(int pg = 0; pg < node->nprograms(); pg++){
 					MaterialNodeRecord rec;
					OPTIX_CHECK(optixSbtRecordPackHeader(materialProgramGroups[node->program() + pg], &rec));
					rec.data = node->sbtData(NodeIndexingInfo{offset_backgroud, offset_backgroud_nodes});
					materialRecords.push_back(rec);
				}
			}
		}

		materialRecordBuffer.alloc_and_upload(materialRecords, stream);

		sbt.callablesRecordBase = materialRecordBuffer.d_pointer();
		sbt.callablesRecordStrideInBytes = sizeof(MaterialNodeRecord);
		sbt.callablesRecordCount = materialRecords.size();
	}

}

void PathTracerState::destroySBT(){
	raygenRecordBuffer.free(stream);
	missRecordBuffer.free(stream);
	hitgroupRecordsBuffer.free(stream);
	materialRecordBuffer.free(stream);
}


void PathTracerState::buildAccel(const std::vector<std::shared_ptr<TriangleMesh>>& meshes){
	std::vector<OptixBuildInput> triangleInput(meshes.size());
	std::vector<std::vector<uint32_t>> triangleInputFlags(meshes.size());
	std::vector<CUdeviceptr> d_vertices(meshes.size());

	for(int i=0; i<meshes.size(); i++){
		const TriangleMesh& mesh = *meshes[i];
		const int materialSize = mesh.materialSlots.size()? mesh.materialSlots.size() : 1;

		d_vertices[i] = mesh.vertexBuffer.d_pointer() + offsetof(Vertex, p);
		triangleInputFlags[i] = std::vector<uint32_t>(materialSize, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);

		triangleInput[i] = {};
			triangleInput[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

			triangleInput[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
			triangleInput[i].triangleArray.numVertices = (int)mesh.vertices.size();
			triangleInput[i].triangleArray.vertexBuffers = &d_vertices[i];
			triangleInput[i].triangleArray.vertexStrideInBytes = sizeof(Vertex);

			triangleInput[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
			triangleInput[i].triangleArray.numIndexTriplets = (int)mesh.indices.size();
			triangleInput[i].triangleArray.indexBuffer = mesh.indexBuffer.d_pointer() + offsetof(Face, vertices);
			triangleInput[i].triangleArray.indexStrideInBytes = sizeof(Face);

			triangleInput[i].triangleArray.flags = triangleInputFlags[i].data();
			triangleInput[i].triangleArray.numSbtRecords = materialSize;
			triangleInput[i].triangleArray.sbtIndexOffsetBuffer = mesh.indexBuffer.d_pointer() + offsetof(Face, material);
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
		tempBuffer.d_pointer(), tempBuffer.size(),
		outputBuffer.d_pointer(), outputBuffer.size(),
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
		asBuffer.d_pointer(), asBuffer.size(),
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

	launchParamsBuffer.alloc(sizeof(LaunchParams), stream);

	std::cout <<"initialized PathTracerState" <<std::endl;
}

void PathTracerState::setScene(const std::shared_ptr<Scene> s){
	scene = s;
	buildAccel(scene->meshes);
	buildSBT();
}

void PathTracerState::removeScene(){
	destroyAccel();
	destroySBT();
	scene = nullptr;
}


PathTracerState::~PathTracerState(){
	removeScene();

	launchParamsBuffer.free(stream);
	cudaEventDestroy(finishEvent);

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

	launchParamsBuffer.upload(&params, 1, stream);

	CUDA_CHECK(cudaEventCreate(&finishEvent));

	OPTIX_CHECK(optixLaunch(
		pipeline,
		stream,
		launchParamsBuffer.d_pointer(),
		launchParamsBuffer.size(),
		&sbt,
		params.image.size.x,
		params.image.size.y,
		1));

	CUDA_CHECK(cudaEventRecord(finishEvent, stream));
}


} // pt5 namespace