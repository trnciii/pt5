#include "scene.hpp"
#include "mesh.hpp"
#include "material/data.h"

namespace pt5{

void SceneBuffer::upload(const Scene& scene, CUstream stream){
	upload_meshes(scene.meshes, stream);
	upload_textures(scene.textures, stream);
	upload_materials(scene.materials, stream);
}

void SceneBuffer::free(CUstream stream){
	free_meshes(stream);
	free_textures(stream);
	free_materials(stream);
};


void SceneBuffer::upload_meshes(const std::vector<TriangleMesh>& meshes, CUstream stream){
	vertexBuffers.resize(meshes.size());
	indexBuffers.resize(meshes.size());
	uvBuffers.resize(meshes.size());

	for(int i=0; i<meshes.size(); i++){
		vertexBuffers[i].alloc_and_upload(meshes[i].vertices, stream);
		indexBuffers[i].alloc_and_upload(meshes[i].indices, stream);
		uvBuffers[i].alloc_and_upload(meshes[i].uv, stream);
	}
}

void SceneBuffer::free_meshes(CUstream stream){
	for(CUDABuffer& buffer : vertexBuffers)buffer.free(stream);
	for(CUDABuffer& buffer : indexBuffers)buffer.free(stream);
	for(CUDABuffer& buffer : uvBuffers)buffer.free(stream);

	cudaStreamSynchronize(stream);
	vertexBuffers.clear();
	indexBuffers.clear();
	uvBuffers.clear();
}


void SceneBuffer::upload_textures(const std::vector<Texture>& s_textures, CUstream stream){
	textures.resize(s_textures.size());
	for(int i=0; i<s_textures.size(); i++)
		textures[i].upload(s_textures[i]);
}

void SceneBuffer::free_textures(CUstream stream){
	for(CUDATexture& texture : textures)texture.free();
	CUDA_SYNC_CHECK();
	textures.clear();
}

void SceneBuffer::upload_materials(const std::vector<std::shared_ptr<Material>>& materials, CUstream stream){
	materialBuffers.resize(materials.size());
	for(int i=0; i<materials.size(); i++){
		const std::shared_ptr<Material>& m = materials[i];
		materialBuffers[i].first.alloc_and_upload(m->ptr(), m->size(), stream);
		materialBuffers[i].second = m->type();
	}

	BSDFData_Diffuse material_default;
	materialBuffer_default.alloc_and_upload(material_default, stream);

	cudaStreamSynchronize(stream);
}

void SceneBuffer::free_materials(CUstream stream){
	for(auto& buffer : materialBuffers)buffer.first.free(stream);
	materialBuffer_default.free(stream);
	cudaStreamSynchronize(stream);
	materialBuffers.clear();
}


}