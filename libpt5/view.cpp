#define GLAD_GL_IMPLEMENTATION
#include "pt5.hpp"

namespace pt5{

View::View(int w, int h)
:width(w), height(h)
{
	CUDA_CHECK(cudaStreamCreate(&stream));

	pixelBuffer.alloc(4*width*height*sizeof(float), stream);
	CUDA_SYNC_CHECK()
}


View::~View(){
	pixelBuffer.free(stream);
	cudaStreamDestroy(stream);
}


uint2 View::size() const{return make_uint2(width, height);}
float4* View::d_pointer() const{return (float4*)pixelBuffer.d_pointer();};


void View::downloadImage(){
	pixels.resize(4*width*height);
	pixelBuffer.download(pixels.data(), pixels.size(), stream);
}


void View::registerGLTexture(GLuint tx){
	glTextureHandle = tx;

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


}
