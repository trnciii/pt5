import bpy
from ... import core


def autoFocalLen(lens, film, x, y):
	if x>y: return 2*lens/film
	else: return (2*lens/film)*(y/x)


def fromObject(bcam):
	camera = core.Camera()

	sx = bcam.data.sensor_width
	sy = bcam.data.sensor_height
	lens = bcam.data.lens
	rx = bpy.context.scene.render.resolution_x
	ry = bpy.context.scene.render.resolution_y

	if bcam.data.sensor_fit == 'HORIZONTAL':
		camera.focalLength = 2*lens/sx
	elif bcam.data.sensor_fit == 'VERTICAL':
		camera.focalLength = (2*lens/sy)*(ry/rx)
	else:
		camera.focalLength = autoFocalLen(lens, sx, rx, ry)


	mat = bcam.matrix_world

	camera.position = mat.to_translation()
	camera.toWorld = mat.to_3x3()

	return camera


def fromView(context, dim):
	camera = core.Camera()
	mat = context.region_data.view_matrix.inverted()
	camera.focalLength = autoFocalLen(context.space_data.lens, 72, *dim)
	camera.position = mat.to_translation()
	camera.toWorld = mat.to_3x3()
	return camera
