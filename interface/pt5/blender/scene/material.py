import bpy
from bpy_extras.node_utils import find_node_input
import numpy as np
import traceback
from ... import core

def getBackground(scene):
	world = scene.world

	if not (world.use_nodes and world.node_tree):
		return  world.color

	output = world.node_tree.get_output_node('CYCLES')
	if not output:
		return world.color

	socket = find_node_input(output, 'Surface')
	filtered = [l.from_node for l in world.node_tree.links if l.to_socket == socket]
	if not (len(filtered)>0 and filtered[0].type == 'BACKGROUND'):
		return  [0,0,0]


	params = filtered[0].inputs
	return np.array(params[0].default_value[:3]) * params[1].default_value



def findImageTexture(tree, socket):
	filtered = [l.from_node for l in tree.links if l.to_socket == socket]
	if not (len(filtered)>0 and filtered[0].type == 'TEX_IMAGE'):
		return None

	image = filtered[0].image
	return core.Texture(np.array(image.pixels).reshape((image.size[1], image.size[0], 4)))


def perseMaterial(mtl, textures):
	if mtl.grease_pencil:
		return core.MTLData_Diffuse([0,0,0], 0)


	if not mtl.use_nodes:
		return core.MTLData_Diffuse(mtl.diffuse_color[:3], 0)


	output = mtl.node_tree.get_output_node('CYCLES')
	if not output:
		return core.MTLData_Diffuse([0,0,0], 0)

	socket = find_node_input(output, 'Surface')
	filtered = [l.from_node for l in mtl.node_tree.links if l.to_socket == socket]
	if not len(filtered) > 0:
		return core.MTLData_Diffuse([0,0,0], 0)


	nodetype = filtered[0].type
	params = filtered[0].inputs

	texture = findImageTexture(mtl.node_tree, params[0])
	tx_index = 0
	if texture:
		textures.append(texture)
		tx_index = len(textures)

	if nodetype == 'EMISSION':
		return core.MTLData_Emission(np.array(params[0].default_value[:3])*params[1].default_value, tx_index)

	elif nodetype == 'BSDF_DIFFUSE':
		return core.MTLData_Diffuse(params[0].default_value[:3], tx_index)

	else:
		return core.MTLData_Diffuse(params[0].default_value[:3], tx_index)



def getMaterials():
	textures = []
	materials = []

	for m in bpy.data.materials.values():
		try:
			materials.append(perseMaterial(m, textures))
		except:
			materials.append(core.MTLData_Emission([1,0,1],0))

			print(m.name)
			traceback.print_exc()

	return materials, textures

