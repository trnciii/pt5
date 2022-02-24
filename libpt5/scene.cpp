#include "scene.hpp"
#include "mesh.hpp"


namespace pt5{

void SceneBuffer::upload(const Scene& scene, CUstream stream){
	upload_images(scene.images);
}

void SceneBuffer::free(CUstream stream){
	free_images();
};


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



}