import pt5

import numpy as np
import matplotlib.pyplot as plt
import bpy
import os, sys, threading


def main():
	scale = bpy.context.scene.render.resolution_percentage/100
	width = int(bpy.context.scene.render.resolution_x*scale)
	height = int(bpy.context.scene.render.resolution_y*scale)


	view = pt5.View(width, height)
	view.clear([0.4, 0.4, 0.4, 0.4])

	if '--background' not in sys.argv:
		window = pt5.Window_py(view)


	pt = pt5.PathTracer()
	pt.setScene(pt5.scene.createSceneFromBlender())

	camera = pt5.scene.getCurrentCameraObject()


	pt.render(view, 1000, camera)
	if not '--background' in sys.argv:
		window.draw(pt)

	pt5.cuda_sync()

	view.downloadImage()
	pixels = np.minimum(1, np.maximum(0, view.pixels**0.4))


	os.makedirs('result', exist_ok=True)
	plt.imsave('result/out_blender.png', pixels)


th = threading.Thread(target=main)
th.start()
th.join()
del th
