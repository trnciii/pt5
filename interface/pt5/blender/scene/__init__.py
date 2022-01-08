import numpy as np
import bpy

from . import camera
from ...core import Image

def create(scene, hide = []):
	from ... import core
	from .material import make_material, perseNodes
	from .object import toTriangleMesh, drawable

	images = [i for i in bpy.data.images.values() if i.name != 'Render Result']

	ret = core.Scene()
	ret.images = [Image(np.array(i.pixels).reshape((i.size[1], i.size[0], 4))) for i in images]

	# materials
	materials = []
	for src in bpy.data.materials.values():
		try:
			materials.append(make_material(perseNodes(src)))
		except:
			materials.append(make_material([core.Emission( ([1,0,1],0), (1, 0) )]))

			print(src.name)
			traceback.print_exc()

	ret.materials = materials

	# background
	try:
		ret.background = make_material(perseNodes(scene.world))
	except:
		ret.background = make_material([core.Background( ([1,0,1],0), (1, 0) )])


	ret.meshes = [m for m in [toTriangleMesh(o) for o in drawable(scene, hide)] if m]

	return ret
