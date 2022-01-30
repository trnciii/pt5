import pt5
from pt5.window import Window

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
	scene.background = pt5.Material([
		pt5.make_node(pt5.Background( ([1,1,1], 0), (1,0) ))
	])


	scene.materials = [
		pt5.Material([
			pt5.make_node(pt5.Mix(1, 2, (0.5, 0))),
			pt5.make_node(pt5.Diffuse( ([0.8, 0.1, 0.1], 0) )),
			pt5.make_node(pt5.Diffuse( ([0.1, 0.9, 0.1], 0) )),
		]),
		pt5.Material([
			pt5.make_node(pt5.Diffuse( ([0.1, 0.8, 0.8], 0) ))
		]),
		pt5.Material([
			pt5.make_node(pt5.Diffuse( ([0.8, 0.5, 0.1], 0) ))
		]),
		pt5.Material([
			pt5.make_node(pt5.Diffuse( ([0.8, 0.3, 0.8], 0) ))
		]),
	]


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


	meshes = [
		pt5.TriangleMesh(
			np.array(
				[(tuple(v), tuple(n)) for v, n in zip(verts0, normals)],
				dtype=pt5.dtype.Vertex_dtype),
			np.array(
				[(tuple(i),tuple(i),	False, m) for i, m in zip(indices, mIDs0)],
				dtype=pt5.dtype.Face_dtype),
			uv0,
			mSlots0),

		pt5.TriangleMesh(
			np.array(
				[(tuple(v), tuple(n)) for v, n in zip(verts1, normals)],
				dtype = pt5.dtype.Vertex_dtype),
			np.array(
				[(tuple(i), tuple(i), False, m) for i, m in zip(indices, mIDs1)],
				dtype = pt5.dtype.Face_dtype),
			uv1,
			mSlots1)
	]

	scene.meshes = meshes



def main(out, background):
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

	if (parent := os.path.dirname(out)) != '':
		os.makedirs(parent, exist_ok=True)
	plt.imsave(out, pt5.linear_to_sRGB(view.pixels))


if ('-o' in sys.argv):
	out = sys.argv[sys.argv.index('-o')+1]
	if os.path.splitext(out)[1] == '':
		out += '.png'
else:
	out = 'result/python.png'

main(out = out, background= '--background' in sys.argv or '-b' in sys.argv)
