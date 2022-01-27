#pragma once

#include <vector>
#include <unordered_map>
#include <string>

#include "vector_math.h"
#include "CUDABuffer.hpp"
#include "material/type.hpp"
#include "mesh.hpp"
#include "sbt.hpp"


namespace pt5{

struct Image{
	uint2 size;
	std::vector<float4> pixels;
};

struct Scene{
	Material background;
	std::vector<TriangleMesh> meshes;
	std::vector<Material> materials;
	std::unordered_map<std::string, Image> images;
};


class SceneBuffer{
	CUstream stream;

	std::vector<CUDABuffer> vertexBuffers;
	std::vector<CUDABuffer> indexBuffers;
	std::vector<CUDABuffer> uvBuffers;

	std::unordered_map<std::string, cudaArray_t> images;

	std::vector<int> offset_material;
	int offset_backgroud;
	std::vector<std::vector<MaterialNodeSBTData>> materialSBTData;
	std::vector<MaterialNodeSBTData> backgroundSBTData;

	void upload_meshes(const std::vector<TriangleMesh>&);
	void free_meshes();

	void upload_images(const std::unordered_map<std::string, Image>& images);
	void free_images();

	void createMaterialData(const std::vector<Material>& materials, const Material& background);
	void destrpyMaterialData();

public:
	SceneBuffer();
	~SceneBuffer();

	void upload(const Scene& scene);
	void free();

	inline void syncStream()const{cudaStreamSynchronize(stream);}

	inline CUdeviceptr vertices(int i) const{return vertexBuffers[i].d_pointer();}
	inline CUdeviceptr indices(int i) const{return indexBuffers[i].d_pointer();}
	inline CUdeviceptr uv(int i) const{return uvBuffers[i].d_pointer();}

	inline int node_output_background()const{return offset_backgroud;}
	inline int node_output_material(int i)const{return offset_material[i];}
	inline MaterialNodeSBTData SBTData_material(int m, int n)const{return materialSBTData[m][n];}
	inline MaterialNodeSBTData SBTData_background(int n)const{return backgroundSBTData[n];}
};


}