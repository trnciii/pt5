import bpy
from bpy_extras.node_utils import find_node_input
import numpy as np
import traceback
from ... import core

def getBackground(world, images):

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
	texture = findImageTexture(world.node_tree, params[0], images)
	if texture:
		return np.array(params[0].default_value[:3]), 0,  params[1].default_value

	else:
		return np.array(params[0].default_value[:3]), 0,  params[1].default_value



def findImageTexture(tree, socket, images):
	filtered = [l.from_node for l in tree.links if l.to_socket == socket]
	if not (len(filtered)>0 and filtered[0].type == 'TEX_IMAGE' and filtered[0].image):
		return None

	node = filtered[0]
	return core.Texture(
		bpy.data.images.values().index(node.image),
		interpolation = node.interpolation,
		extension = node.extension)


def perseMaterial(mtl, images):

	if mtl.grease_pencil:
		return [core.Diffuse( ([0,0,0], 0) )]


	if not mtl.use_nodes:
		return [core.Diffuse( (mtl.diffuse_color[:3], 0) )]


	output = mtl.node_tree.get_output_node('CYCLES')
	if not output:
		return [core.Diffuse([0,0,0], 0)]

	socket = find_node_input(output, 'Surface')
	filtered = [l.from_node for l in mtl.node_tree.links if l.to_socket == socket]
	if not len(filtered) > 0:
		return [core.Diffuse( ([0,0,0], 0) )]


	nodetype = filtered[0].type
	params = filtered[0].inputs

	texture = findImageTexture(mtl.node_tree, params[0], images)
	if texture:
		if nodetype == 'EMISSION':
			return [core.Emission( (params[0].default_value[:3], 1), (params[1].default_value, 0) ), texture]

		elif nodetype == 'BSDF_DIFFUSE':
			print(texture)
			return [core.Diffuse( (params[0].default_value[:3], 1) ), texture]

		else:
				return [core.Diffuse( (params[0].default_value[:3], 1) ), texture]


	else:
		if nodetype == 'EMISSION':
			return [core.Emission( (params[0].default_value[:3], 0), (params[1].default_value, 0) )]

		elif nodetype == 'BSDF_DIFFUSE':
			return [core.Diffuse( (params[0].default_value[:3], 0) )]

		else:
			return [core.Diffuse( (params[0].default_value[:3], 0) )]



def getMaterials(scene, images):
	materials = []

	for m_bl in bpy.data.materials.values():
		try:
			materials.append(core.Material([core.make_node(data) for data in perseMaterial(m_bl, images)]))
		except:
			materials.append(core.Material([core.make_node(core.Emission( ([1,0,1],0), (1, 0) ))]))

			print(m_bl.name)
			traceback.print_exc()

	return materials

