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
	std::vector<CUDABuffer> materialBuffers;

	void upload_meshes(const std::vector<TriangleMesh>&, CUstream);
	void upload_textures(const std::vector<Texture>&, CUstream);
	void upload_materials(const std::vector<Material>&, CUstream);

	void free_meshes(CUstream stream);
	void free_textures(CUstream stream);
	void free_materials(CUstream stream);

public:

	void upload(const Scene& scene, CUstream stream);
	void free(CUstream stream);

	inline CUdeviceptr vertices(int i) const{return vertexBuffers[i].d_pointer();}
	inline CUdeviceptr indices(int i) const{return indexBuffers[i].d_pointer();}
	inline CUdeviceptr uv(int i) const{return uvBuffers[i].d_pointer();}
	inline CUdeviceptr materials(int i) const{return materialBuffers[i].d_pointer();}
};


}