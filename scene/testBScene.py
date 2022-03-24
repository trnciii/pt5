import pt5

import numpy as np
import bpy
import threading
import sys, os


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


def main(out):
	scene = bpy.context.scene
	scale = bpy.context.scene.render.resolution_percentage/100
	width = int(bpy.context.scene.render.resolution_x*scale)
	height = int(bpy.context.scene.render.resolution_y*scale)

	view = pt5.View(width, height)
	view.clear([0.4, 0.4, 0.4, 0.4])


	pt = pt5.PathTracer()
	pt.setScene(pt5.scene.create(scene))

	camera = pt5.scene.camera.fromObject(scene.camera)


	pt.render(view, 1000, camera)
	pt.waitForRendering()

	view.downloadImage()

	image = addImage('pt5 result', view.pixels)
	image.save_render(out)


if ('-o' in sys.argv):
	out = sys.argv[sys.argv.index('-o')+1]
	if os.path.splitext(out)[1] == '':
		out += '.png'
else:
	out = 'result/blender_script.png'

th = threading.Thread(target=main, kwargs={'out':out})
th.start()
th.join()
del th
