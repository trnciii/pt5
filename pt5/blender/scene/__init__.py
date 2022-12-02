import numpy as np
import bpy
import traceback
import time

from . import camera
from ...core import Image

def make_image(src):
	image = Image(*src.size)
	src.pixels.foreach_get(np.array(image, copy=False).ravel())
	return image

def is_used(img, candidates):
	for mat in candidates:
		if mat.user_of_id(img): return True
	return False


def create(scene, hide = []):
	from ... import core
	from .material import make_material, parseNodes
	from .object import toTriangleMesh, drawable

	ret = core.Scene()

	images = {img.name : make_image(img)
		for img in bpy.data.images
		if is_used(img, bpy.data.materials.values() + bpy.data.worlds.values())
	}

	for v in images.values():
		v.upload()

	# materials
	materials = []
	for src in bpy.data.materials.values():
		try:
			nodes = parseNodes(src, images)
			materials.append(make_material(nodes))
		except:
			materials.append(make_material([core.Emission( ([1,0,1],0), (1, 0) )]))
			print(src.name)
			traceback.print_exc()

	ret.materials = materials

	# background
	try:
		nodes = parseNodes(scene.world, images)
		ret.background = make_material(parseNodes(scene.world, images))
	except:
		ret.background = make_material([core.Background( ([1,0,1],0), (1, 0) )])
		traceback.print_exc()


	ret.meshes = [m for m in [toTriangleMesh(o) for o in drawable(scene, hide)] if m]

	return ret
