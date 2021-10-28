#define GLAD_GL_IMPLEMENTATION
#include "pt5.hpp"

namespace pt5{

View::View(int w, int h)
:width(w), height(h)
{
	CUDA_CHECK(cudaStreamCreate(&stream));

	pixels.resize(4*width*height);

	pixelBuffer.alloc(4*width*height*sizeof(float), stream);
	CUDA_SYNC_CHECK()

	std::cout <<"created View" <<std::endl;
}


View::~View(){
	if(hasGLTexture()) destroyGLTexture();
	pixelBuffer.free(stream);
	cudaStreamDestroy(stream);
	std::cout <<"destructed View" <<std::endl;
}


void View::downloadImage(){
	pixels.resize(4*width*height);
	pixelBuffer.download(pixels.data(), pixels.size(), stream);
}


void View::createGLTexture(){
	glGenTextures(1, &glTextureHandle);

	glBindTexture(GL_TEXTURE_2D, glTextureHandle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	// synchronous
	CUDA_CHECK(cudaGraphicsGLRegisterImage(
		&cudaTextureResourceHandle,
		glTextureHandle,
		GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsWriteDiscard));

	std::cout <<"created and registered opengl texture" <<std::endl;
}

void View::destroyGLTexture(){
	assert(hasGLTexture());

	CUDA_CHECK(cudaGraphicsUnregisterResource(cudaTextureResourceHandle))
	glDeleteTextures(1, &glTextureHandle);
	glTextureHandle = 0;
	cudaTextureResourceHandle = nullptr;

	std::cout <<"removed opengl texture" <<std::endl;
}

bool View::hasGLTexture(){
	return glTextureHandle != 0 && cudaTextureResourceHandle != nullptr;
}

void View::updateGLTexture(){
	cudaArray* texture_ptr;
	CUDA_CHECK(cudaGraphicsMapResources(1, &cudaTextureResourceHandle, stream));
	CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cudaTextureResourceHandle, 0, 0));

	CUDA_CHECK(cudaMemcpy2DToArrayAsync(
		texture_ptr,
		0, 0,
		(void*)pixelBuffer.d_pointer(),
		4*width*sizeof(float),
		4*width*sizeof(float),
		height,
		cudaMemcpyDeviceToDevice,
		stream
		))

	CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaTextureResourceHandle, stream));
}

void View::clear(float4 c){
	std::fill_n((float4*)pixels.data(), width*height, c);
	pixelBuffer.upload(pixels.data(), pixels.size(), stream);
	if(glTextureHandle) updateGLTexture();
}


}
