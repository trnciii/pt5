#define GLAD_GL_IMPLEMENTATION
#include "pt5.hpp"

namespace pt5{

View::View(int w, int h)
:width(w), height(h)
{
	if(!glfwInit()) assert(0);

	window = glfwCreateWindow(width, height, "pt5 view", NULL, NULL);
	if (!window){
			assert(0);
	    glfwTerminate();
	}

	glfwMakeContextCurrent(window);

	gladLoadGL(glfwGetProcAddress);



	CUDA_CHECK(cudaStreamCreate(&stream));

	pixelBuffer.alloc(4*width*height*sizeof(float), stream);
	CUDA_SYNC_CHECK()


	glGenTextures(1, &glTextureHandle);
	glBindTexture(GL_TEXTURE_2D, glTextureHandle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);


	CUDA_CHECK(cudaGraphicsGLRegisterImage(
		&cudaTextureResourceHandle,
		glTextureHandle,
		GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsWriteDiscard));

	CUDA_SYNC_CHECK()

}


View::~View(){
	pixelBuffer.free(stream);
	cudaStreamDestroy(stream);

	glDeleteTextures(1, &glTextureHandle);
	glfwDestroyWindow(window);
	glfwTerminate();
}

void View::downloadImage(){
	pixels.resize(4*width*height);
	pixelBuffer.download(pixels.data(), pixels.size(), stream);
}


void View::updateTexture(){
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


void View::drawWindow(){
	glEnable(GL_FRAMEBUFFER_SRGB);

	glViewport(0,0,width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, (float)width, 0, (float)height, -1, 1);


	do{
		updateTexture();

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, glTextureHandle);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glBegin(GL_QUADS);
		{
			glTexCoord2f(0.f, 0.f);
			glVertex3f(0.f, (float)height, 0.f);

			glTexCoord2f(0.f, 1.f);
			glVertex3f(0.f, 0.f, 0.f);

			glTexCoord2f(1.f, 1.f);
			glVertex3f((float)width, 0.f, 0.f);

			glTexCoord2f(1.f, 0.f);
			glVertex3f((float)width, (float)height, 0.f);
		}
		glEnd();


		glfwSwapBuffers(window);
    glfwPollEvents();
	}while(
		!glfwWindowShouldClose(window)
		&& (cudaEventQuery(*tracerFinishEvent) == cudaErrorNotReady)
		);
}


}
