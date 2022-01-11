import bpy
import bmesh
import numpy as np
from ... import core
from ... import dtype
import traceback


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


def drawable(scene, hide = []):
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
			dtype=dtype.Vertex_dtype
		)

		faces = np.array([(
			tuple(p.vertices[:3]),
			tuple(p.loop_indices[:3]),
			(p.use_smooth),
			(p.material_index))
			for p in mesh.polygons],
			dtype=dtype.Face_dtype
		)

		if mesh.uv_layers.active:
			uv = np.array(
				[data.uv for data in mesh.uv_layers.active.data])
		else:
			uv = np.array(
				[[0,0] for i in range(sum([p.loop_total for p in mesh.polygons]))])

		mtls = [bpy.data.materials.find(k) for k in mesh.materials.keys()]

		return core.TriangleMesh(verts, faces, uv, mtls)

	except:
		print(obj.name)
		traceback.print_exc()
		return None