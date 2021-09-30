import bpy
import bmesh
import numpy as np
from . import core
import traceback


def setCamera(scene):
	camera = bpy.context.scene.camera
	camera.data.sensor_fit = 'HORIZONTAL'
	mat = camera.matrix_world

	print(mat)

	scene.camera.focalLength = 2*camera.data.lens/camera.data.sensor_width
	scene.camera.position = mat.to_translation()
	scene.camera.toWorld = mat.to_3x3()

	print('focal', scene.camera.focalLength)


def setBackground(scene):
    tree = bpy.data.worlds['World'].node_tree

    for l in tree.links:
        to = l.to_node
        frm = l.from_node
        if type(to) is bpy.types.ShaderNodeOutputWorld:
            if to.is_active_output and l.to_socket.name == 'Surface':
                scene.background = frm.inputs[0].default_value[:3]

def setMaterials(scene):
  materials = []

  for m in bpy.data.materials.values():

    links = m.node_tree.links
    nodes = m.node_tree.nodes

    for l in links:
      t = l.to_node
      f = l.from_node
      if type(t) is bpy.types.ShaderNodeOutputMaterial:
        if t.is_active_output and l.to_socket.name == 'Surface':
          m_pt5 = core.Material()
          m_pt5.color = f.inputs[0].default_value[:3]
          materials.append(m_pt5)

  scene.materials = materials


def getTriangles(obj):
	depsgraph = bpy.context.evaluated_depsgraph_get()
	object_eval = obj.evaluated_get(depsgraph)

	mesh = bpy.data.meshes.new_from_object(object_eval)

	bm = bmesh.new()
	bm.from_mesh(mesh)

	bmesh.ops.triangulate(bm, faces=bm.faces[:])

	bm.to_mesh(mesh)
	bm.free

	return mesh


def setMesh(scene):
	meshes = []
	for obj in bpy.data.objects.values():
		try:
			mesh = getTriangles(obj)
			mat = obj.matrix_world

			verts = [mat@v.co for v in mesh.vertices]
			normals = [(mat@v.normal - mat.to_translation()).normalized() for v in mesh.vertices]
			indices = [p.vertices[:3] for p in mesh.polygons]
			mtlIDs = [p.material_index for p in mesh.polygons]
			mtlSlots = [bpy.data.materials.find(k) for k in mesh.materials.keys()]

			meshes.append(core.createTriangleMesh(verts, normals, indices, mtlIDs, mtlSlots))
		except:
			traceback.print_exc()

	scene.meshes = meshes


def createScene(scene):

	setCamera(scene)
	setBackground(scene)
	setMaterials(scene)
	setMesh(scene)
