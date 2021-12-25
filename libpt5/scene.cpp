#include "scene.hpp"
#include "mesh.hpp"
#include "texture.hpp"


namespace pt5{

void SceneBuffer::upload(const Scene& scene, CUstream stream){
	upload_meshes(scene.meshes, stream);
	upload_images(scene.images);
	create_textures(scene.textures, scene, stream);
}

void SceneBuffer::free(CUstream stream){
	free_meshes(stream);
	destroy_textures(stream);
	free_images();
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



void SceneBuffer::upload_images(const std::vector<Image>& s_images){
	images.resize(s_images.size());
	for(int i=0; i<s_images.size(); i++){
		const Image& image = s_images[i];
		cudaArray_t& array = images[i];

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
	for(cudaArray_t& array : images) CUDA_CHECK(cudaFreeArray(array));
}



void SceneBuffer::create_textures(const std::vector<Texture>& s_textures, const Scene& scene, CUstream stream){
	textures.resize(s_textures.size());
	for(int i=0; i<s_textures.size(); i++){
		const Texture& t = s_textures[i];
		cudaResourceDesc resDesc = {};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = images[t.image];
		CUDA_CHECK(cudaCreateTextureObject(&textures[i], &resDesc, &t.desc, nullptr));
	}
}

void SceneBuffer::destroy_textures(CUstream stream){
	for(cudaTextureObject_t& t : textures)
		CUDA_CHECK(cudaDestroyTextureObject(t));
	textures.clear();
}


}