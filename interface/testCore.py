import pt5

import matplotlib.pyplot as plt
import numpy as np
import os, sys


def createScene(scene, camera):

	# camera
	camera.position = [0, -25, 1]
	camera.focalLength = 2
	camera.toWorld = [[1, 0, 0],
										[0, 0,-1],
										[0, 1, 0]]

	# background
	scene.background = [0.8, 0.8, 0.8]

	# material
	materials = [pt5.Material() for i in range(3)]
	materials[0].albedo = [0.8, 0.8, 0.3]
	materials[1].albedo = [0.1, 0.8, 0.8]
	materials[2].albedo = [0.8, 0.5, 0.1]

	m2 = pt5.Material()
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
		pt5.createTriangleMesh(verts0, normals, indices, mIDs0, mSlots0),
		pt5.createTriangleMesh(verts1, normals, indices, mIDs1, mSlots1)
	]

	scene.meshes = meshes



def main(background, use_python_window):
	w, h = 1200, 800

	view = pt5.View(w,h)

	if not background:
		window = pt5.Window_py(view) if use_python_window else pt5.Window_cpp(view)
		view.clear([0.3, 0.3, 0.3, 1])


	scene = pt5.Scene()
	camera = pt5.Camera()
	createScene(scene, camera)

	pt = pt5.PathTracer()
	pt.setScene(scene)

	pt.render(view, 1000, camera)
	if not background:
		window.draw(pt)

	pt5.cuda_sync()

	view.downloadImage()

	os.makedirs('result', exist_ok=True)
	plt.imsave('result/out_py.png', view.pixels)

	# explicitly destroy window before view.
	# (view could be deleted first when using c++ wrapped Window class...)
	del window
	del view

for i in range(1):
	main(background='--background' in sys.argv, use_python_window=True)
	print('-'*40)
	main(background='--background' in sys.argv, use_python_window=False)
	print('-'*40)
