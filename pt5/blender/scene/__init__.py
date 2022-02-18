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


def create(scene, hide = []):
	from ... import core
	from .material import make_material, parseNodes
	from .object import toTriangleMesh, drawable

	ret = core.Scene()


	# materials
	images = set()
	materials = []
	for src in bpy.data.materials.values():
		try:
			nodes = parseNodes(src)
			images.update({n.image for n in nodes if isinstance(n, core.Texture)})
			materials.append(make_material(nodes))
		except:
			materials.append(make_material([core.Emission( ([1,0,1],0), (1, 0) )]))
			print(src.name)
			traceback.print_exc()

	ret.materials = materials

	# background
	try:
		nodes = parseNodes(scene.world)
		images.update({n.image for n in nodes if isinstance(n, core.Texture)})
		ret.background = make_material(parseNodes(scene.world))
	except:
		ret.background = make_material([core.Background( ([1,0,1],0), (1, 0) )])


	ret.meshes = [m for m in [toTriangleMesh(o) for o in drawable(scene, hide)] if m]

	ret.images = {k : make_image(bpy.data.images[k]) for k in images}


	return ret
