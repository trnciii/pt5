#pragma once

#include <stdint.h>
#include <vector>
#include "vector_math.h"
#include "CUDABuffer.hpp"
#include "mesh.hpp"

namespace pt5{

struct Material;

struct Scene{
	float3 background = {0.4, 0.4, 0.4};
	std::vector<TriangleMesh> meshes;
	std::vector<Material> materials;
};


class SceneBuffer{
	std::vector<CUDABuffer> vertexBuffers;
	std::vector<CUDABuffer> indexBuffers;
	std::vector<CUDABuffer> uvBuffers;

public:
	~SceneBuffer(){assert(allocated() == 0);}

	inline uint32_t allocated() const{
		assert((vertexBuffers.size() == indexBuffers.size())
			&& (vertexBuffers.size() == uvBuffers.size()));
		return vertexBuffers.size();
	}

	void upload(const Scene& scene, CUstream stream){
		vertexBuffers.resize(scene.meshes.size());
		indexBuffers.resize(scene.meshes.size());
		uvBuffers.resize(scene.meshes.size());

		for(int i=0; i<scene.meshes.size(); i++){
			vertexBuffers[i].alloc_and_upload(scene.meshes[i].vertices, stream);
			indexBuffers[i].alloc_and_upload(scene.meshes[i].indices, stream);
			uvBuffers[i].alloc_and_upload(scene.meshes[i].uv, stream);
		}

		cudaStreamSynchronize(stream);
	}

	void free(CUstream stream){
		for(CUDABuffer& buffer : vertexBuffers)buffer.free(stream);
		for(CUDABuffer& buffer : indexBuffers)buffer.free(stream);
		for(CUDABuffer& buffer : uvBuffers)buffer.free(stream);

		cudaStreamSynchronize(stream);
		vertexBuffers.clear();
		indexBuffers.clear();
		uvBuffers.clear();
	};

	inline CUdeviceptr vertices(int i) const{return vertexBuffers[i].d_pointer();}
	inline CUdeviceptr indices(int i) const{return indexBuffers[i].d_pointer();}
	inline CUdeviceptr uv(int i) const{return uvBuffers[i].d_pointer();}

};


}