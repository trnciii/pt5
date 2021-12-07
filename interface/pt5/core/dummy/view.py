import numpy as np
from OpenGL.GL import *

class View:
	def __init__(self, w, h):
		self.x, self.y = w, h
		self.glTextureHandle = None

	def downloadImage(self): pass

	def createGLTexture(self):
		data = self.pixels
		data[:, :, 3] = 0.5

		self.glTextureHandle = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.glTextureHandle);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.x, self.y, 0, GL_RGBA, GL_FLOAT, data);
		glBindTexture(GL_TEXTURE_2D, 0);

	def destroyGLTexture(self):
		glBindTexture(GL_TEXTURE_2D, 0)
		glDeleteTextures(1, [self.glTextureHandle])
		self.glTextureHandle = None

	def updateGLTexture(self): pass

	def clear(self, c): pass

	@property
	def GLTexture(self):
		return self.glTextureHandle

	@property
	def hasGLTexture(self):
		return self.glTextureHandle is not None

	@property
	def size(self):
		return self.x, self.y

	@property
	def pixels(self):
		return np.array([0.8, 0.2, 0.4, 1]*(self.x*self.y)).reshape((self.y, self.x, 4))
