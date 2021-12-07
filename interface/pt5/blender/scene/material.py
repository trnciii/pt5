import bpy
from bpy_extras.node_utils import find_node_input
import numpy as np
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


def create_material_diffuse(x, y, z):
	m = core.Material()
	m.emission = 0,0,0
	m.albedo = x,y,z
	return m

def create_material_emit(x,y,z, strength = 1):
	m = core.Material()
	m.emission = x, y, z
	m.emission *= strength
	m.albedo = 0,0,0
	return m


def perseMaterial(mtl):
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

	if nodetype == 'EMISSION':
		return create_material_emit(*params[0].default_value[:3],	params[1].default_value)

	elif nodetype == 'BSDF_DIFFUSE':
		return create_material_diffuse(*params[0].default_value[:3])

	else:
		return create_material_diffuse(*params[0].default_value[:3])



def getMaterials():
	materials = []

	for m in bpy.data.materials.values():
		try:
			materials.append(perseMaterial(m))
		except:
			magenta = core.Material()
			magenta.albedo
			magenta.emission = [1,0,1]
			materials.append(magenta)

			print(m.name)
			traceback.print_exc()

	return materials

