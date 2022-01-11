def nodeProgramNames():pass
def setNodeIndices():pass

class MaterialNode:pass

class Material:
	def __init__(self, *args):
		self.nodes = []

class Mix:
	def __init__(self, *args):
		self.shader1 = 0
		self.shader2 = 0
		self.factor = (0, 0)

class Diffuse:
	def __init__(self, *args):
		self.color = ([0,0,0], 0)

class Emission:
	def __init__(self, *args):
		self.color = ([0,0,0], 0)
		self.strength = (0,0)

class Texture:
	def __init__(self, *args, **kwargs):
		self.image = args[0]

class Background:
	def __init__(self, *args):
		self.color = ([0,0,0], 0)
		self.strength = (0,0)

def make_node(data):return MaterialNode()