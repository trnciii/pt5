import numpy as np

class Scene:
	def __init__(self):
		self.materials = []
		self.meshes = []
		self._background = [0,0,0]

	def upload(self):pass
	def free(self):pass

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


class TriangleMesh:
	def __init__(self, v, f, uv, m):
		self.vertices = v
		self.indices = f
		self.uv = uv
		self.materialSlots = m
