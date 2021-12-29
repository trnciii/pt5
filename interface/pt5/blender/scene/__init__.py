import numpy as np
import bpy

from . import camera
from ...core import Image

def create(scene, hide = []):
	from ... import core
	from .material import getBackground, getMaterials
	from .object import toTriangleMesh, drawable

	images = [i for i in bpy.data.images.values() if i.name != 'Render Result']

	ret = core.Scene()
	ret.images = [Image(np.array(i.pixels).reshape((i.size[1], i.size[0], 4))) for i in images]
	ret.materials = getMaterials(scene)
	ret.background = getBackground(scene.world, images)
	ret.meshes = [m for m in [toTriangleMesh(o) for o in drawable(scene, hide)] if m]
	return ret
