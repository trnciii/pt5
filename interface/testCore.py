import pt5
from pt5.window import Window as Window

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

	uv0 = np.array([
		[0.0, 1.0],
		[0.0, 0.5],
		[0.5, 0.5],
		[0.5, 1.0],
		[1.0, 1.0],
		[1.0, 0.5]
	])

	uv1 = uv0 - [0, 0.5]

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


	scene.meshes = [
		pt5.TriangleMesh(
			np.array(
				[(tuple(v), tuple(n)) for v, n in zip(verts0, normals)],
				dtype=pt5.Vertex_dtype),
			np.array(
				[(tuple(i),tuple(i),	False, m) for i, m in zip(indices, mIDs0)],
				dtype=pt5.Face_dtype),
			uv0,
			mSlots0),

		pt5.TriangleMesh(
			np.array(
				[(tuple(v), tuple(n)) for v, n in zip(verts1, normals)],
				dtype = pt5.Vertex_dtype),
			np.array(
				[(tuple(i), tuple(i), False, m) for i, m in zip(indices, mIDs1)],
				dtype = pt5.Face_dtype),
			uv1,
			mSlots1)
	]

	scene.upload()



def main(background):
	w, h = 1200, 800

	view = pt5.View(w,h)

	if not background:
		window = Window(view)
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
	plt.imsave('result/out_py.png', np.maximum(0, np.minimum(1, view.pixels)))


main(background='--background' in sys.argv)
