import numpy as np
import bpy

from . import camera
from ...core import Image

def create(scene, hide = []):
	from ... import core
	from .material import make_material, perseNodes
	from .object import toTriangleMesh, drawable

	ret = core.Scene()


	# materials
	images = []
	materials = []
	for src in bpy.data.materials.values():
		try:
			nodes = perseNodes(src)
			images += [n.image for n in nodes if isinstance(n, core.Texture)]
			materials.append(make_material(nodes))
		except:
			materials.append(make_material([core.Emission( ([1,0,1],0), (1, 0) )]))

			print(src.name)
			traceback.print_exc()

	ret.materials = materials

	# background
	try:
		nodes = perseNodes(scene.world)
		images += [n.image for n in nodes if isinstance(n, core.Texture)]
		ret.background = make_material(perseNodes(scene.world))
	except:
		ret.background = make_material([core.Background( ([1,0,1],0), (1, 0) )])


	ret.meshes = [m for m in [toTriangleMesh(o) for o in drawable(scene, hide)] if m]

	ret.images = {
		k : Image(np.array(v.pixels).reshape((v.size[1], v.size[0], 4)))
		for k, v in {k:bpy.data.images[k] for k in images}.items()
	}


	return ret
