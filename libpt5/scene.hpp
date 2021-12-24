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
	std::vector<Material> materials;
	std::vector<Texture> textures;
	std::vector<Image> images;
};


class SceneBuffer{
	std::vector<CUDABuffer> vertexBuffers;
	std::vector<CUDABuffer> indexBuffers;
	std::vector<CUDABuffer> uvBuffers;

	std::vector<cudaArray_t> images;
	std::vector<cudaTextureObject_t> textures;


	void upload_meshes(const std::vector<TriangleMesh>&, CUstream);
	void upload_images(const std::vector<Image>& images);
	void create_textures(const std::vector<Texture>& s_textures, const Scene& scene, CUstream stream);

	void free_meshes(CUstream stream);
	void free_images();
	void destroy_textures(CUstream stream);


public:
	void upload(const Scene& scene, CUstream stream);
	void free(CUstream stream);

	inline CUdeviceptr vertices(int i) const{return vertexBuffers[i].d_pointer();}
	inline CUdeviceptr indices(int i) const{return indexBuffers[i].d_pointer();}
	inline CUdeviceptr uv(int i) const{return uvBuffers[i].d_pointer();}

};


}