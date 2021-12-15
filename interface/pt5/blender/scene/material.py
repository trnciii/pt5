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


def create_material_diffuse(x, y, z, texture = 0):
	m = core.MTLData_Diffuse()
	m.color = x,y,z
	m.texture = texture
	return m

def create_material_emit(x,y,z, strength = 1, texture = 0):
	m = core.MTLData_Emission()
	m.color = x, y, z
	m.color *= strength
	m.texture = texture
	return m


def findImageTexture(tree, socket):
	filtered = [l.from_node for l in tree.links if l.to_socket == socket]
	if not (len(filtered)>0 and filtered[0].type == 'TEX_IMAGE'):
		return None

	image = filtered[0].image
	return core.Texture(np.array(image.pixels).reshape((image.size[1], image.size[0], 4)))


def perseMaterial(mtl, textures):
	if mtl.grease_pencil:
		return create_material_diffuse(0,0,0)


	if not mtl.use_nodes:
		return create_material_diffuse(*mtl.diffuse_color[:3])


	output = mtl.node_tree.get_output_node('CYCLES')
	if not output:
		return create_material_diffuse(0,0,0)

	socket = find_node_input(output, 'Surface')
	filtered = [l.from_node for l in mtl.node_tree.links if l.to_socket == socket]
	if not len(filtered) > 0:
		return create_material_diffuse(0,0,0)


	nodetype = filtered[0].type
	params = filtered[0].inputs

	texture = findImageTexture(mtl.node_tree, params[0])
	tx_index = 0
	if texture:
		textures.append(texture)
		tx_index = len(textures)

	if nodetype == 'EMISSION':
		return create_material_emit(
			*params[0].default_value[:3],
			strength = params[1].default_value,
			texture = tx_index)

	elif nodetype == 'BSDF_DIFFUSE':
		return create_material_diffuse(
			*params[0].default_value[:3],
			texture = tx_index)

	else:
		return create_material_diffuse(
			*params[0].default_value[:3],
			texture = tx_index)



def getMaterials():
	textures = []
	materials = []

	for m in bpy.data.materials.values():
		try:
			materials.append(perseMaterial(m, textures))
		except:
			materials.append(create_material_emit(1,0,1))

			print(m.name)
			traceback.print_exc()

	return materials, textures

