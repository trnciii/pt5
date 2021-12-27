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


class Prop:
	def __init__(self, d, s):
		self.default = d
		self.sub = s

	def __str__(self):
		if self.default != None:
			return '(default: ' + str(self.default) +', input: ' + str(self.sub) + ')'
		else:
			return '(input: ' + str(self.sub) + ')'

	def nodeindex(self, nodes):
		input_id = nodes.index(self.sub.node) if self.sub else 0
		return self.default, input_id


class Graph:
	def __init__(self, node, props):
		self.node = node
		self.props = props

	def __str__(self):
		return self.node.name + str({str(k):str(v) for k, v in self.props.items()})

	def nodes(self):
		nodes = [self.node]
		for p in self.props.values():
			if p.sub != None:
				nodes += p.sub.nodes()

		return nodes



def constructNodeTree(node, tree):

	def find_socket_input(socket):
		filtered = [l.from_node for l in tree.links if l.to_socket == socket]
		if(len(filtered)>0): return Graph(*constructNodeTree(filtered[0], tree))
		else: return None

	inputs = node.inputs

	if node.type == 'OUTPUT_MATERIAL':
		return node, {
			'Surface': Prop(None, find_socket_input(inputs['Surface'])),
		}

	if node.type == 'EMISSION':
		return node,{
			'Color': Prop(inputs['Color'].default_value[:3], find_socket_input(inputs['Color'])),
			'Strength': Prop(inputs['Strength'].default_value, find_socket_input(inputs['Strength'])),
		}


	if node.type == 'BSDF_DIFFUSE':
		return node, {
			'Color': Prop(inputs['Color'].default_value[:3], find_socket_input(inputs['Color'])),
		}


	if node.type == 'MIX_SHADER':
		return node,{
			'Fac': Prop(inputs['Fac'].default_value, find_socket_input(inputs['Fac'])),
			'shader 1': Prop(None, find_socket_input(inputs[1])),
			'shader 2': Prop(None, find_socket_input(inputs[2])),
		}


	if node.type == 'TEX_IMAGE':
		return node, {}


	if node.type == 'BSDF_PRINCIPLED': # read as diffuse
		return node, {
			'Base Color': Prop(inputs['Base Color'].default_value[:3], find_socket_input(inputs['Base Color'])),
		}



def compatible(mtl):
	if mtl.grease_pencil: return False
	if mtl.use_nodes and not mtl.node_tree.get_output_node('CYCLES'): return False
	return True


def perseMaterial(mtl, images):

	if not compatible(mtl): return [core.Diffuse((0,0,0),0)]

	if not mtl.use_nodes:
		return [core.Diffuse( (mtl.diffuse_color[:3], 0) )]


	tree = mtl.node_tree
	output = tree.get_output_node('CYCLES')


	graph = constructNodeTree(output, tree)[1]['Surface']

	nodes = graph.sub.nodes()
	nodes = sorted(set(nodes), key = nodes.index)


	# print()
	# print('#'*40)
	# print(mtl.name)
	# print('graph', graph)
	# print('nodes', [n.name for n in nodes])


	# for n in nodes:
	# 	print('  ', nodes.index(n), n.name, end=': ')

	# 	props = constructNodeTree(n, tree)[1]
	# 	print([(k, nodes.index(v.sub.node) if v.sub else 0) for k, v in props.items()])

	# print('-'*40)

	ret = []

	for n in nodes:
		inputs = n.inputs
		props = constructNodeTree(n, tree)[1]

		if n.type == 'EMISSION':
			ret.append(core.Emission(
				props['Color'].nodeindex(nodes),
				props['Strength'].nodeindex(nodes)
			))

		if n.type == 'BSDF_DIFFUSE':
			ret.append(core.Diffuse(props['Color'].nodeindex(nodes)))

		if n.type == 'MIX_SHADER':
			ret.append(core.Mix(
				props['shader 1'].nodeindex(nodes)[1],
				props['shader 2'].nodeindex(nodes)[1],
				props['Fac'].nodeindex(nodes),
			))


		if n.type == 'TEX_IMAGE':
			ret.append(core.Texture(
				bpy.data.images.values().index(n.image),
				interpolation = n.interpolation,
				extension = n.extension
			))

		if n.type == 'BSDF_PRINCIPLED': # read as diffuse
			ret.append(core.Diffuse(props['Base Color'].nodeindex(nodes)))

	return ret


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

