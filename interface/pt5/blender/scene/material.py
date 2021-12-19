import bpy
from bpy_extras.node_utils import find_node_input
import numpy as np
import traceback
from ... import core

def getBackground(world, textures, images):

	if not (world.use_nodes and world.node_tree):
		return  world.color, 0, 1

	output = world.node_tree.get_output_node('CYCLES')
	if not output:
		return world.color, 0, 1

	socket = find_node_input(output, 'Surface')
	filtered = [l.from_node for l in world.node_tree.links if l.to_socket == socket]
	if not (len(filtered)>0 and filtered[0].type == 'BACKGROUND'):
		return  [0,0,0], 0, 1


	params = filtered[0].inputs
	image, texture = findImageTexture(world.node_tree, params[0], images)
	tx_index = 0
	if texture:
		textures.append(texture)
		tx_index = len(textures)

	return np.array(params[0].default_value[:3]), tx_index,  params[1].default_value



def findImageTexture(tree, socket, images):
	filtered = [l.from_node for l in tree.links if l.to_socket == socket]
	if not (len(filtered)>0 and filtered[0].type == 'TEX_IMAGE' and filtered[0].image):
		return None, None

	node = filtered[0]
	image = node.image
	return image.name, core.Texture(
		bpy.data.images.values().index(image),
		interpolation = node.interpolation,
		extension = node.extension)


def perseMaterial(mtl, textures, images):

	if mtl.grease_pencil:
		return core.BSDF_Diffuse([0,0,0], 0)


	if not mtl.use_nodes:
		return core.BSDF_Diffuse(mtl.diffuse_color[:3], 0)


	output = mtl.node_tree.get_output_node('CYCLES')
	if not output:
		return core.BSDF_Diffuse([0,0,0], 0)

	socket = find_node_input(output, 'Surface')
	filtered = [l.from_node for l in mtl.node_tree.links if l.to_socket == socket]
	if not len(filtered) > 0:
		return core.BSDF_Diffuse([0,0,0], 0)


	nodetype = filtered[0].type
	params = filtered[0].inputs

	image, texture = findImageTexture(mtl.node_tree, params[0], images)
	tx_index = 0
	if texture:
		textures.append(texture)
		tx_index = len(textures)

	if nodetype == 'EMISSION':
		return core.BSDF_Emission(np.array(params[0].default_value[:3]), tx_index, params[1].default_value, 0)

	elif nodetype == 'BSDF_DIFFUSE':
		return core.BSDF_Diffuse(params[0].default_value[:3], tx_index)

	else:
		return core.BSDF_Diffuse(params[0].default_value[:3], tx_index)



def getMaterials(scene, images):
	textures = []
	materials = []

	for m_bl in bpy.data.materials.values():
		try:
			materials.append(perseMaterial(m_bl, textures, images))
		except:
			materials.append(core.BSDF_Emission([1,0,1],0, 1, 0))

			print(m_bl.name)
			traceback.print_exc()

	world = getBackground(scene.world, textures, images)

	return materials, textures, world

