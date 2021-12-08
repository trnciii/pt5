#pragma once

#include <stdint.h>
#include <vector>
#include "vector_math.h"
#include "CUDABuffer.hpp"
#include "mesh.hpp"
#include "texture.hpp"
#include "material.h"

namespace pt5{


struct Scene{
	float3 background = {0.4, 0.4, 0.4};
	std::vector<TriangleMesh> meshes;
	std::vector<Material> materials;
	std::vector<Texture> textures;
};


class SceneBuffer{
	std::vector<CUDABuffer> vertexBuffers;
	std::vector<CUDABuffer> indexBuffers;
	std::vector<CUDABuffer> uvBuffers;
	std::vector<CUDATexture> textures;
	CUDABuffer materialBuffer;

public:
	~SceneBuffer(){assert(allocated() == 0);}

	inline uint32_t allocated() const{
		assert((vertexBuffers.size() == indexBuffers.size())
			&& (vertexBuffers.size() == uvBuffers.size()));
		return vertexBuffers.size();
	}


	void upload(const Scene& scene, CUstream stream){
		{ // mesh
			vertexBuffers.resize(scene.meshes.size());
			indexBuffers.resize(scene.meshes.size());
			uvBuffers.resize(scene.meshes.size());

			for(int i=0; i<scene.meshes.size(); i++){
				vertexBuffers[i].alloc_and_upload(scene.meshes[i].vertices, stream);
				indexBuffers[i].alloc_and_upload(scene.meshes[i].indices, stream);
				uvBuffers[i].alloc_and_upload(scene.meshes[i].uv, stream);
			}
		}

		{ // texture
			textures.resize(scene.textures.size());

			for(int i=0; i<scene.textures.size(); i++)
				textures[i].upload(scene.textures[i]);
		}

		{ // material
			materialBuffer.alloc_and_upload(scene.materials, stream);
		}

		cudaStreamSynchronize(stream);
	}

	void free(CUstream stream){
		{ // mesh
			for(CUDABuffer& buffer : vertexBuffers)buffer.free(stream);
			for(CUDABuffer& buffer : indexBuffers)buffer.free(stream);
			for(CUDABuffer& buffer : uvBuffers)buffer.free(stream);

			cudaStreamSynchronize(stream);
			vertexBuffers.clear();
			indexBuffers.clear();
			uvBuffers.clear();
		}

		{ // texture
			for(CUDATexture& texture : textures)texture.free();
			CUDA_SYNC_CHECK();
			textures.clear();
		}

		{	// material
			materialBuffer.free(stream);
			cudaStreamSynchronize(stream);
		}
	};

	inline CUdeviceptr vertices(int i) const{return vertexBuffers[i].d_pointer();}
	inline CUdeviceptr indices(int i) const{return indexBuffers[i].d_pointer();}
	inline CUdeviceptr uv(int i) const{return uvBuffers[i].d_pointer();}
	inline CUdeviceptr materials(int i) const{return materialBuffer.d_pointer() + i*sizeof(Material);}

};


}