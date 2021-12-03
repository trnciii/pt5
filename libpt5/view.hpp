#pragma once

#include <cuda_gl_interop.h>
#include <vector>

#include "CUDABuffer.hpp"


namespace pt5{


class View{
public:
	View(int w, int h);
	~View();

	uint2 size() const{return make_uint2(width, height);}
	float4* bufferPtr() const{return (float4*)pixelBuffer.d_pointer();}

	void createGLTexture();
	void updateGLTexture();
	void destroyGLTexture();
	bool hasGLTexture();
	GLuint GLTexture()const{return glTextureHandle;};

	void downloadImage();
	void clear(float4 c);

	std::vector<float> pixels;

private:
	CUstream stream;

	int width;
	int height;

	cudaGraphicsResource* cudaTextureResourceHandle = nullptr;
	GLuint glTextureHandle = GL_FALSE;
	CUDABuffer pixelBuffer;
};


}