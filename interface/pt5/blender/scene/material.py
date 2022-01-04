import bpy
from bpy_extras.node_utils import find_node_input
import numpy as np
import traceback
from ... import core


class Prop:
	def __init__(self, d, i):
		self.default = d
		self.input = i

	def __str__(self):
		if self.default != None:
			return '(default: ' + str(self.default) +', input: ' + str(self.input) + ')'
		else:
			return '(input: ' + str(self.input) + ')'

	def nodeindex(self, nodes):
		input_id = nodes.index(self.input.node) if self.input else 0
		return self.default, input_id


class Graph:
	def __init__(self, node, tree):

		self.node = node

		def find_socket_input(socket):
			filtered = [l.from_node for l in tree.links if l.to_socket == socket]
			if(len(filtered)>0): return Graph(filtered[0], tree)
			else: return None

		inputs = node.inputs

		if node.type == 'OUTPUT_MATERIAL':
			self.props = {
				'Surface': Prop(None, find_socket_input(inputs['Surface'])),
			}


		elif node.type == 'OUTPUT_WORLD':
			self.props = {
				'Surface': Prop(None, find_socket_input(inputs['Surface'])),
			}


		elif node.type == 'EMISSION':
			self.props = {
				'Color': Prop(inputs['Color'].default_value[:3], find_socket_input(inputs['Color'])),
				'Strength': Prop(inputs['Strength'].default_value, find_socket_input(inputs['Strength'])),
			}

			self.create = lambda nodes: core.Emission(
				self.props['Color'].nodeindex(nodes),
				self.props['Strength'].nodeindex(nodes)
			)



		elif node.type == 'BSDF_DIFFUSE':
			self.props = {
				'Color': Prop(inputs['Color'].default_value[:3], find_socket_input(inputs['Color'])),
			}

			self.create = lambda nodes: core.Diffuse(self.props['Color'].nodeindex(nodes))


		elif node.type == 'MIX_SHADER':
			self.props = {
				'Fac': Prop(inputs['Fac'].default_value, find_socket_input(inputs['Fac'])),
				'shader 1': Prop(None, find_socket_input(inputs[1])),
				'shader 2': Prop(None, find_socket_input(inputs[2])),
			}

			self.create = lambda nodes: core.Mix(
				self.props['shader 1'].nodeindex(nodes)[1],
				self.props['shader 2'].nodeindex(nodes)[1],
				self.props['Fac'].nodeindex(nodes),
			)


		elif node.type == 'TEX_IMAGE':
			self.props = {}
			self.create = lambda nodes: core.Texture(
				bpy.data.images.values().index(node.image),
				interpolation = node.interpolation,
				extension = node.extension
			)


		elif node.type == 'BSDF_PRINCIPLED': # read as diffuse
			self.props = {
				'Base Color': Prop(inputs['Base Color'].default_value[:3], find_socket_input(inputs['Base Color'])),
			}

			self.create = lambda nodes: core.Diffuse(self.props['Base Color'].nodeindex(nodes))


		elif node.type == 'BACKGROUND':
			self.props = {
				'Color': Prop(inputs['Color'].default_value[:3], find_socket_input(inputs['Color'])),
				'Strength': Prop(inputs['Strength'].default_value, find_socket_input(inputs['Strength']))
			}

			self.create = lambda nodes: core.Background(
				self.props['Color'].nodeindex(nodes),
				self.props['Strength'].nodeindex(nodes)
			)

		elif node.type == 'TEX_ENVIRONMENT':
			self.props = {}
			self.create = lambda nodes: core.Environment(
				bpy.data.images.values().index(node.image)
			)

		else:
			print('failed to perse a', node.type, 'node ', node.name)


	def __str__(self):
		return self.node.name + str({str(k):str(v) for k, v in self.props.items()})

	def nodes(self):
		nodes = [self.node]
		for p in self.props.values():
			if p.input != None:
				nodes += p.input.nodes()

		return nodes



def compatible(mtl):
	if isinstance(mtl, bpy.types.Material):
		if mtl.grease_pencil:
			return False
		if mtl.use_nodes and not mtl.node_tree.get_output_node('CYCLES'):
			return False
		return True

	elif isinstance(mtl, bpy.types.World):
		if mtl.use_nodes and not mtl.node_tree.get_output_node('CYCLES'):
			return False
		return True


def perseMaterialNodes(mtl):
	if not compatible(mtl):
		return [core.Diffuse((0,0,0),0)]

	if not mtl.use_nodes:
		return [core.Diffuse( (mtl.diffuse_color[:3], 0) )]


	tree = mtl.node_tree
	output = tree.get_output_node('CYCLES')
	if not output:
		return [core.Background( ([0,0,0], 0), (1,0) )]


	graph = Graph(output, tree).props['Surface']

	nodes = graph.input.nodes()
	nodes = sorted(set(nodes), key = nodes.index)

	# print()
	# print('#'*40)
	# print(mtl.name)
	# print('graph', graph)
	# print('nodes', [n.name for n in nodes])

	# for n in nodes:
	# 	print('  ', nodes.index(n), n.name, end=': ')

	# 	props = Graph(n, tree).props
	# 	print([(k, v.default, nodes.index(v.input.node) if v.input else 0) for k, v in props.items()])

	# print('-'*40)

	return [Graph(n, tree).create(nodes) for n in nodes]


def make_material(nodes):
	return core.Material([core.make_node(data) for data in nodes])


def getBackground(world):
	try:
		if not compatible(world):
			return [core.Background( ([1,1,1], 0), (1,0) )]

		if not world.use_nodes:
			return  [core.Background( (world.color[:3], 0), (1,0) )]

		tree = world.node_tree
		output = tree.get_output_node('CYCLES')
		if not output:
			return [core.Background( ([0,0,0], 0), (1,0) )]

		graph = Graph(output, tree).props['Surface']

		nodes = graph.input.nodes()
		nodes = sorted(set(nodes), key = nodes.index)

		return [Graph(n, tree).create(nodes) for n in nodes]

	except:
		return [core.Background( ([1,0,1],0), (1, 0) )]



def getMaterials(scene):
	materials = []

	for m_bl in bpy.data.materials.values():
		try:
			materials.append(make_material(perseMaterialNodes(m_bl)))
		except:
			materials.append(make_material([core.Emission( ([1,0,1],0), (1, 0) )]))

			print(m_bl.name)
			traceback.print_exc()

	return materials

