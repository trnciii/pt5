#include "pt5.hpp"

namespace pt5{

View::View(int w, int h)
:width(w), height(h)
{
	CUDA_CHECK(cudaStreamCreate(&stream));

	pixelBuffer.alloc(4*width*height*sizeof(float), stream);

	CUDA_SYNC_CHECK();
}

View::~View(){
	pixelBuffer.free(stream);
	CUDA_SYNC_CHECK();
}

void View::downloadImage(){
	pixels.resize(4*width*height);
	pixelBuffer.download(pixels.data(), pixels.size(), stream);
}

void View::initWindow(){
	if(!glfwInit()) assert(0);

	window = glfwCreateWindow(width, height, "pt5 view", NULL, NULL);
	if (!window){
			assert(0);
	    glfwTerminate();
	}

	glfwMakeContextCurrent(window);


	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);

	glViewport(0,0,width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, (float)width, 0, (float)height, -1, 1);
}


void View::showWindow(const cudaEvent_t* const finished){
	initWindow();
	do{
		downloadImage();


		GLuint fbTexture {0};
		glGenTextures(1, &fbTexture);
		glBindTexture(GL_TEXTURE_2D, fbTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, pixels.data());

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, fbTexture);
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
		&& (cudaEventQuery(*finished) == cudaErrorNotReady)
		);
}

}
