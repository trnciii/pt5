from pt5 import BScene, core
import numpy as np
import matplotlib.pyplot as plt
import bpy
import os, sys


def main():
	scale = bpy.context.scene.render.resolution_percentage/100
	width = int(bpy.context.scene.render.resolution_x*scale)
	height = int(bpy.context.scene.render.resolution_y*scale)


	view = core.View(width, height)

	if '--background' not in sys.argv:
		window = core.Window(view)


	scene = core.Scene()
	BScene.createScene(scene)

	pt = core.PathTracer()
	pt.init()
	pt.setScene(scene)
	pt.initLaunchParams(view, 1000)


	if '--background' in sys.argv:
		pt.render()
	else:
		pt.render()
		window.draw(view, pt)

	core.cuda_sync()

	view.downloadImage()
	pixels = np.array(np.minimum(1, np.maximum(0, view.pixels**0.4)))


	os.makedirs('result', exist_ok=True)
	plt.imsave('result/out_blender.png', pixels)

main()
print('end')