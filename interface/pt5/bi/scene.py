import bpy
import bmesh
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


def setBackground(scene):
	tree = bpy.data.worlds['World'].node_tree

	for l in tree.links:
		to = l.to_node
		frm = l.from_node
		if type(to) is bpy.types.ShaderNodeOutputWorld:
			if to.is_active_output and l.to_socket.name == 'Surface':
				scene.background = frm.inputs[0].default_value[:3]


def perseMaterial(mtl):
	if mtl.node_tree and mtl.grease_pencil == None:
		for l in mtl.node_tree.links:
			if type(l.to_node) is bpy.types.ShaderNodeOutputMaterial:
				if l.to_node.is_active_output and l.to_socket.name == 'Surface':
					shader_node = l.from_node

					m_pt5 = core.Material()

					if shader_node.type == 'EMISSION':
						m_pt5.albedo = [0,0,0]
						m_pt5.emission = shader_node.inputs[0].default_value[:3]
						m_pt5.emission *= shader_node.inputs[1].default_value

					elif shader_node.type == 'BSDF_DIFFUSE':
						m_pt5.albedo = shader_node.inputs[0].default_value[:3]
						m_pt5.emission = [0,0,0]

					else:
						m_pt5.albedo = shader_node.inputs[0].default_value[:3]
						m_pt5.emission = [0,0,0]

					return m_pt5

	black = core.Material()
	black.albedo = [0,0,0]
	black.emission = [0,0,0]
	return black


def setMaterials(scene):
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

	scene.materials = materials


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

		return mesh

	except:
		print('error: failed to evaluate object as mesh')
		traceback.print_exc()
		return None


def geometries():
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

	return [o for o in bpy.data.objects.values()
		if o.type in types
		and not (o.type=='META' and '.' in o.name)
	]


def setObjects(scene):
	meshes = []

	for obj in geometries():
		try:
			mesh = getTriangles(obj)
			if mesh and len(mesh.polygons)>0:
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

				meshes.append(core.TriangleMesh(verts, faces, mtls))

		except:
			print(obj.name)
			traceback.print_exc()

	scene.meshes = meshes


def createSceneFromBlender():
	scene = core.Scene()
	setBackground(scene)
	setMaterials(scene)
	setObjects(scene)
	return scene
