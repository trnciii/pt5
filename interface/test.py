from pt5 import core

import matplotlib.pyplot as plt
import numpy as np


def createScene(scene):

	# camera
	scene.camera.position = [0, -25, 1]
	scene.camera.focalLength = 2
	scene.camera.toWorld = [[1, 0, 0],
													[0, 0,-1],
													[0, 1, 0]]

	# background
	scene.background = [0.2, 0.1, 0.4]

	# material
	materials = [core.Material() for i in range(3)]
	materials[0].color = [0.8, 0.8, 0.3]
	materials[1].color = [0.1, 0.8, 0.8]
	materials[2].color = [0.8, 0.5, 0.1]

	m2 = core.Material()
	m2.color = [0.8, 0.3, 0.8]
	materials.append(m2)

	scene.materials = materials

	# mesh 0
	verts0 = [
		[-4, 0, 6],
		[-4, 0, 2],
		[ 0, 0, 2],
		[ 0, 0, 6],
		[ 4, 0, 6],
		[ 4, 0, 2]
	]

	normals0 = [
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0]
	]

	indices0 = [
		[0, 1, 2],
		[2, 3, 0],
		[3, 2, 5],
		[5, 4, 3]
	]

	mIDs0 = [1, 1,	0, 0]
	mSlots0 = [0, 3]

	# mesh 1
	verts1 = [
		[-4, 0, 6-6],
		[-4, 0, 2-6],
		[ 0, 0, 2-6],
		[ 0, 0, 6-6],
		[ 4, 0, 6-6],
		[ 4, 0, 2-6]
	]

	normals1 = [
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0]
	]

	indices1 = [
		[0, 1, 2],
		[2, 3, 0],
		[3, 2, 5],
		[5, 4, 3]
	]

	mIDs1 = [1, 2, 0, 3]
	mSlots1 = [0,1,2,3]


	meshes = [
		core.createTriangleMesh(verts0, normals0, indices0, mIDs0, mSlots0),
		core.createTriangleMesh(verts1, normals1, indices1, mIDs1, mSlots1)
	]

	scene.meshes = meshes




w, h = 1200, 800

scene = core.Scene()
createScene(scene)

print(scene.camera.toWorld)

pt = core.PathTracer()
pt.init()
pt.setScene(scene)
pt.initLaunchParams(w, h)

pt.render()

pixels4 = np.array(pt.pixels()).reshape((h,w,4))

plt.imshow(pixels4)
plt.show()

plt.imsave('out_py.png', pixels4)