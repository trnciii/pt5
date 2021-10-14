from pt5 import core

import matplotlib.pyplot as plt
import numpy as np
import os, sys


def createScene(scene):

	# camera
	scene.camera.position = [0, -25, 1]
	scene.camera.focalLength = 2
	scene.camera.toWorld = [[1, 0, 0],
													[0, 0,-1],
													[0, 1, 0]]

	# background
	scene.background = [0.8, 0.8, 0.8]

	# material
	materials = [core.Material() for i in range(3)]
	materials[0].albedo = [0.8, 0.8, 0.3]
	materials[1].albedo = [0.1, 0.8, 0.8]
	materials[2].albedo = [0.8, 0.5, 0.1]

	m2 = core.Material()
	m2.albedo = [0.8, 0.3, 0.8]
	materials.append(m2)

	scene.materials = materials

	verts0 = np.array([
		[-4, 0, 6],
		[-4, 0, 2],
		[ 0, 0, 2],
		[ 0, 0, 6],
		[ 4, 0, 6],
		[ 4, 0, 2]
	])

	verts1 = verts0 - [0,0,6]

	normals = [
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0],
		[0, -1, 0]
	]

	indices = [
		[0, 1, 2],
		[2, 3, 0],
		[3, 2, 5],
		[5, 4, 3]
	]

	mIDs0 = [1, 1,	0, 0]
	mSlots0 = [0, 3]


	mIDs1 = [1, 2, 0, 3]
	mSlots1 = [0,1,2,3]


	meshes = [
		core.createTriangleMesh(verts0, normals, indices, mIDs0, mSlots0),
		core.createTriangleMesh(verts1, normals, indices, mIDs1, mSlots1)
	]

	scene.meshes = meshes


def main():
	w, h = 1200, 800

	view = core.View(w,h)

	if '--background' not in sys.argv:
		window = core.Window(view)


	scene = core.Scene()
	createScene(scene)

	pt = core.PathTracer()
	pt.init()
	pt.setScene(scene)
	pt.initLaunchParams(view, 100)


	if '--background' in sys.argv:
		pt.render()
	else:
		window.draw(view, pt)

	core.cuda_sync()


	view.downloadImage()

	os.makedirs('result', exist_ok=True)
	plt.imsave('result/out_py.png', view.pixels)


main()