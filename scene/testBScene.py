import pt5

import numpy as np
import bpy
import threading


def addImage(name, data):
	images = bpy.data.images

	data = np.maximum(0, np.minimum(1, data))

	if name in images:
		images.remove(images[name])

	im = images.new(name = name, width = data.shape[1], height = data.shape[0])
	im.colorspace_settings.name = 'Linear'
	im.use_view_as_render = True
	im.pixels = np.flipud(data).flatten()

	return im


def main():
	scene = bpy.context.scene
	scale = scene.render.resolution_percentage/100
	width = int(scene.render.resolution_x*scale)
	height = int(scene.render.resolution_y*scale)


	view = pt5.View(width, height)
	view.clear([0.4, 0.4, 0.4, 0.4])


	pt = pt5.PathTracer()
	pt.setScene(pt5.scene.createSceneFromBlender(scene))

	camera = pt5.scene.createCameraFromObject(scene.camera)


	pt.render(view, 1000, camera)
	pt5.cuda_sync()

	view.downloadImage()

	image = addImage('pt5 result', view.pixels)
	image.save_render('result/out_blender.png')


th = threading.Thread(target=main)
th.start()
th.join()
del th
