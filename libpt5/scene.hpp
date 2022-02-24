#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

#include "vector_math.h"
#include "CUDABuffer.hpp"
#include "material/type.hpp"
#include "mesh.hpp"


namespace pt5{

struct Image{
	uint2 size;
	std::vector<float4> pixels;
};

struct Scene{
	Material background;
	std::vector<std::shared_ptr<TriangleMesh>> meshes;
	std::vector<Material> materials;
	std::unordered_map<std::string, Image> images;
};


class SceneBuffer{
	std::unordered_map<std::string, cudaArray_t> images;

	void upload_images(const std::unordered_map<std::string, Image>& images);
	void free_images();

public:
	void upload(const Scene& scene, CUstream stream);
	void free(CUstream stream);

	inline const std::unordered_map<std::string, cudaArray_t>& get_images()const{return images;}
};


}