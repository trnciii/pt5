#include "scene.hpp"
#include "mesh.hpp"

#include <iostream>

namespace pt5{

SceneBuffer::SceneBuffer(){
	CUDA_CHECK(cudaStreamCreate(&stream));
}

SceneBuffer::~SceneBuffer(){
	cudaStreamDestroy(stream);
}

void SceneBuffer::upload(const Scene& scene){
	upload_meshes(scene.meshes);
	upload_images(scene.images);
	createMaterialData(scene.materials, scene.background);
}

void SceneBuffer::free(){
	free_meshes();
	free_images();
	destrpyMaterialData();
};


void SceneBuffer::upload_meshes(const std::vector<TriangleMesh>& meshes){
	vertexBuffers.resize(meshes.size());
	indexBuffers.resize(meshes.size());
	uvBuffers.resize(meshes.size());

	for(int i=0; i<meshes.size(); i++){
		vertexBuffers[i].alloc_and_upload(meshes[i].vertices, stream);
		indexBuffers[i].alloc_and_upload(meshes[i].indices, stream);
		uvBuffers[i].alloc_and_upload(meshes[i].uv, stream);
	}
}

void SceneBuffer::free_meshes(){
	for(CUDABuffer& buffer : vertexBuffers)buffer.free(stream);
	for(CUDABuffer& buffer : indexBuffers)buffer.free(stream);
	for(CUDABuffer& buffer : uvBuffers)buffer.free(stream);

	cudaStreamSynchronize(stream);
	vertexBuffers.clear();
	indexBuffers.clear();
	uvBuffers.clear();
}



void SceneBuffer::upload_images(const std::unordered_map<std::string, Image>& s_images){
	for(const auto& [k, image] : s_images){
		cudaArray_t& array = images[k];
		cudaChannelFormatDesc channelFormatDesc = cudaCreateChannelDesc<float4>();

		uint32_t pitch = image.size.x * 4 * sizeof(float);
		CUDA_CHECK(cudaMallocArray(&array, &channelFormatDesc, image.size.x, image.size.y));
		CUDA_CHECK(cudaMemcpy2DToArray(
			array,
			0, 0,
			image.pixels.data(),
			pitch, pitch, image.size.y,
			cudaMemcpyHostToDevice));

	}
}

void SceneBuffer::free_images(){
	for(auto& [k, array] : images) CUDA_CHECK(cudaFreeArray(array));
	images.clear();
}



void SceneBuffer::createMaterialData(const std::vector<Material>& materials, const Material& background){

	offset_material.resize(materials.size());
	offset_material[0] = 0;
	for(int i=1; i<materials.size(); i++)
		offset_material[i] = offset_material[i-1] + materials[i-1].nprograms();

	offset_default_diffuse = offset_material.back() + materials.back().nprograms();
	offset_backgroud = offset_default_diffuse + 3;


	// create sbt data
	materialSBTData.resize(materials.size() + 1);

	for(int i=0; i<materials.size(); i++){
		const int offset = offset_material[i];
		const Material& material = materials[i];

		const auto offset_nodes = material.offset_nodes();
		for(const auto& node : material.nodes){
			materialSBTData[i].emplace_back(node->sbtData(
				NodeIndexingInfo{offset, offset_nodes, images}
			));
		}
	}

	{
		int i = materials.size();
		const int offset = offset_backgroud;
		const Material& material = background;

		const auto offset_nodes = material.offset_nodes();
		for(const auto& node : material.nodes){
			materialSBTData[i].emplace_back(node->sbtData(
				NodeIndexingInfo{offset, offset_nodes, images}
			));
		}
	}

}

void SceneBuffer::destrpyMaterialData(){
	materialSBTData.clear();
	offset_material.clear();
}


}