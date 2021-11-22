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
		glDeleteTextures(1, self.glTextureHandle)
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


class PathTracer:
	def __init__(self):pass
	def setScene(self, s):pass
	def removeScene(self):pass
	def render(self, view, spp, camera):pass

	@property
	def running(self):
		return True


class Scene:
	def __init__(self):
		self.materials = []
		self.meshes = []
		self._background = [0,0,0,0]

	@property
	def background(self):
		return self._background

	@background.setter
	def background(self, c):
		self._background = np.array(c)


class Camera:
	def __init__(self):
		self.focalLength = 2

	@property
	def position(self):
		return self._position

	@position.setter
	def position(self, p):
		self._position = np.array(p)

	@property
	def toWorld(self):
		return self._toWorld

	@toWorld.setter
	def toWorld(self, m):
		self._toWorld = np.array(m).reshape((3,3))


class Material:
	def __init__(self):
		self._albedo = np.array([0,0,0])
		self._emission = np.array([0,0,0])

	@property
	def albedo(self):
		return self._albedo

	@albedo.setter
	def albedo(self, a):
		self._albedo = np.array(a)

	@property
	def emission(self):
		return self._emission

	@emission.setter
	def emission(self, e):
		self._emission = np.array(e)


float3_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
uint3_dtype = [('x', '<i4'), ('y', '<i4'), ('z', '<i4')]

Vertex_dtype = [('p', float3_dtype), ('n', float3_dtype)]
Face_dtype = [
	('vertices', uint3_dtype),
	('smooth', 'i1'),
	('material', '<i4')
]

class TriangleMesh:
	def __init__(self, *args):pass

def cuda_sync():pass
