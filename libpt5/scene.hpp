#pragma once

#include <vector>
#include <memory>
#include "vector_math.h"
#include "CUDABuffer.hpp"
#include "texture.hpp"
#include "sbt.hpp"
#include "material/type.hpp"

namespace pt5{

struct TriangleMesh;

struct Scene{
	struct{
		float3 color = {0.4, 0.4, 0.4};
		uint32_t texture = 0;
		float strength = 1;
	}background;
	std::vector<TriangleMesh> meshes;
	std::vector<std::shared_ptr<Material>> materials;
	std::vector<Texture> textures;
};


class SceneBuffer{
	std::vector<CUDABuffer> vertexBuffers;
	std::vector<CUDABuffer> indexBuffers;
	std::vector<CUDABuffer> uvBuffers;
	std::vector<CUDATexture> textures;
	std::vector<std::pair<CUDABuffer, MaterialType>> materialBuffers;
	CUDABuffer materialBuffer_default;

	void upload_meshes(const std::vector<TriangleMesh>&, CUstream);
	void upload_textures(const std::vector<Texture>&, CUstream);
	void upload_materials(const std::vector<std::shared_ptr<Material>>& materials, CUstream stream);

	void free_meshes(CUstream stream);
	void free_textures(CUstream stream);
	void free_materials(CUstream stream);

public:

	void upload(const Scene& scene, CUstream stream);
	void free(CUstream stream);

	inline CUdeviceptr vertices(int i) const{return vertexBuffers[i].d_pointer();}
	inline CUdeviceptr indices(int i) const{return indexBuffers[i].d_pointer();}
	inline CUdeviceptr uv(int i) const{return uvBuffers[i].d_pointer();}

	inline CUdeviceptr materials(int i) const{
		if(0<=i && i<materialBuffers.size())
			return materialBuffers[i].first.d_pointer();
		else
			return materialBuffer_default.d_pointer();
	}

	inline uint32_t materialTypeIndex(int i) const{
		if(0<=i && i<materialBuffers.size())
			return static_cast<int>(materialBuffers[i].second);
		else
			return static_cast<int>(MaterialType::Diffuse);
	}

};


}