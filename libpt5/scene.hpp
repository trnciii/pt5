#pragma once

#include <vector>
#include "vector_math.h"
#include "CUDABuffer.hpp"
#include "material/type.hpp"


namespace pt5{

struct TriangleMesh;

struct Image{
	uint2 size;
	std::vector<float4> pixels;
};

struct Scene{
	struct{
		float3 color = {0.4, 0.4, 0.4};
		uint32_t texture = 0;
		float strength = 1;
	}background;
	std::vector<TriangleMesh> meshes;
	std::vector<Material> materials;
	std::vector<Image> images;
};


class SceneBuffer{
	std::vector<CUDABuffer> vertexBuffers;
	std::vector<CUDABuffer> indexBuffers;
	std::vector<CUDABuffer> uvBuffers;

	std::vector<cudaArray_t> images;


	void upload_meshes(const std::vector<TriangleMesh>&, CUstream);
	void free_meshes(CUstream stream);

	void upload_images(const std::vector<Image>& images);
	void free_images();


public:
	void upload(const Scene& scene, CUstream stream);
	void free(CUstream stream);

	inline CUdeviceptr vertices(int i) const{return vertexBuffers[i].d_pointer();}
	inline CUdeviceptr indices(int i) const{return indexBuffers[i].d_pointer();}
	inline CUdeviceptr uv(int i) const{return uvBuffers[i].d_pointer();}

	inline const std::vector<cudaArray_t>& get_images()const{return images;}
};


}