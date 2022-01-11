import numpy as np

class Scene:
	def __init__(self):
		self.materials = []
		self.meshes = []
		self.images = {}
		self.background = None


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


class TriangleMesh:
	def __init__(self, v, f, uv, m):pass


class Image(np.ndarray):
	def __new__(self, *args):
		if len(args) == 1:
			print('one', *args)
			return args[0].reshape( (*args[0].shape, 4) )

		elif len(args) == 2:
			print('two', *args)
			return np.zeros((args[1], args[0], 4), dtype = np.float32)

		elif len(args) == 3:
			print('three', *args)
			return args[2].reshape((args[1], args[0], 4))



