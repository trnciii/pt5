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
	inline CUdeviceptr materials(int i) const{return materialBuffers[i].first.d_pointer();}
	inline CUdeviceptr material_default() const{return materialBuffer_default.d_pointer();}
	inline BSDF materialData(int i)const{
		int offset = 3*static_cast<int>(materialBuffers[i].second);
		return (BSDF){materials(i), offset, offset+1, offset+2};
	}
	inline BSDF materialData_default()const{
		return (BSDF){material_default(), 0, 1, 2};
	}
};


}