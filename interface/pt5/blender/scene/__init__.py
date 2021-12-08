from . import camera


def create(scene, hide = []):
	from ... import core
	from .material import getBackground, getMaterials
	from .object import toTriangleMesh, drawable

	ret = core.Scene()
	ret.background = getBackground(scene)
	ret.materials, ret.textures = getMaterials()
	ret.meshes = [m for m in [toTriangleMesh(o) for o in drawable(scene, hide)] if m]
	return ret
