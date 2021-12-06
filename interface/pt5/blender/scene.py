import bpy
import bmesh
from bpy_extras.node_utils import find_node_input

import numpy as np
from .. import core
import traceback
import time


def autoFocalLen(lens, film, x, y):
	if x>y: return 2*lens/film
	else: return (2*lens/film)*(y/x)


def createCameraFromObject(bcam):
	camera = core.Camera()

	sx = bcam.data.sensor_width
	sy = bcam.data.sensor_height
	lens = bcam.data.lens
	rx = bpy.context.scene.render.resolution_x
	ry = bpy.context.scene.render.resolution_y

	if bcam.data.sensor_fit == 'HORIZONTAL':
		camera.focalLength = 2*lens/sx
	elif bcam.data.sensor_fit == 'VERTICAL':
		camera.focalLength = (2*lens/sy)*(ry/rx)
	else:
		camera.focalLength = autoFocalLen(lens, sx, rx, ry)


	mat = bcam.matrix_world

	camera.position = mat.to_translation()
	camera.toWorld = mat.to_3x3()

	return camera


def getViewAsCamera(context, dim):
	camera = core.Camera()
	mat = context.region_data.view_matrix.inverted()
	camera.focalLength = autoFocalLen(context.space_data.lens, 72, *dim)
	camera.position = mat.to_translation()
	camera.toWorld = mat.to_3x3()
	return camera


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


def getTriangles(obj):
	try:
		depsgraph = bpy.context.evaluated_depsgraph_get()
		object_eval = obj.evaluated_get(depsgraph)

		mesh = object_eval.to_mesh()

		bm = bmesh.new()
		bm.from_mesh(mesh)

		bmesh.ops.triangulate(bm, faces=bm.faces[:])

		bm.to_mesh(mesh)
		bm.free()

		if len(mesh.polygons)>0: return mesh
		else: return None

	except:
		print('error: failed to evaluate object as mesh')
		traceback.print_exc()
		return None


def drawable(scene, hide = None):
	types = [
		'MESH',
		'CURVE',
		'SURFACE',
		'META',
		'FONT',
		# 'HAIR',
		# 'POINTCLOUD',
		# 'VOLUME',
		# 'GPENCIL',
		# 'ARMATURE',
		# 'LATTICE',
		# 'EMPTY',
		# 'LIGHT',
		# 'LIGHT_PROBE',
		# 'CAMERA',
		# 'SPEAKER'
	]

	return [o for o in scene.objects.values()
		if o.type in types
		and (o not in hide)
		and not (o.type=='META' and '.' in o.name)
	]


def toTriangleMesh(obj):
	try:
		mesh = getTriangles(obj)

		if not mesh: return None

		mat = obj.matrix_world
		verts = np.array([(
				tuple(mat@v.co),
				tuple((mat@v.normal - mat.to_translation()).normalized()))
				for v in mesh.vertices
			],
			dtype=core.Vertex_dtype
		)

		faces = np.array([(
				tuple(p.vertices[:3]),
				(p.use_smooth),
				(p.material_index))
				for p in mesh.polygons
			],
			dtype=core.Face_dtype
		)

		mtls = [bpy.data.materials.find(k) for k in mesh.materials.keys()]

		return core.TriangleMesh(verts, faces, mtls)

	except:
		print(obj.name)
		traceback.print_exc()
		return None


def createSceneFromBlender(scene, hide = None):
	ret = core.Scene()
	ret.background = getBackground(scene)
	ret.materials = getMaterials()
	ret.meshes = [m for m in [toTriangleMesh(o) for o in drawable(scene, hide)] if m]
	return ret
