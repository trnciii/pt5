from pt5 import core

import matplotlib.pyplot as plt
import numpy as np
import os, sys
import glfw
from OpenGL.GL import *


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


class Window:
	def __init__(self, view):
		self.view = view


		if not glfw.init():
			self.use = False
			return

		self.size = view.size
		self.window = glfw.create_window(self.size[0], self.size[1], "python window", None, None)
		if not self.window:
			glfw.terminate()
			self.use = False
			return

		self.use = True


		glfw.make_context_current(self.window)

		glEnable(GL_FRAMEBUFFER_SRGB)
		glViewport(0,0,self.size[0], self.size[1])
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0, self.size[0], 0, self.size[1], -1, 1)

		glEnable(GL_TEXTURE_2D)
		self.tx = glGenTextures(1)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
		self.view.registerGLTexture(self.tx)


	def __del__(self):
		glDeleteTextures(1, self.tx)
		glfw.destroy_window(self.window)
		glfw.terminate()


	@property
	def avairable(self):
		return self.use


	def draw(self, pt):
		if not self.use:
			core.cuda_sync()
			return

		glBindTexture(GL_TEXTURE_2D, self.tx)


		while not glfw.window_should_close(self.window) and pt.running:
			self.view.updateGLTexture()

			glBegin(GL_QUADS)
			glTexCoord2f(0, 0)
			glVertex3f(0, self.size[1], 0)

			glTexCoord2f(0, 1)
			glVertex3f(0, 0, 0)

			glTexCoord2f(1, 1)
			glVertex3f(self.size[0], 0, 0)

			glTexCoord2f(1, 0)
			glVertex3f(self.size[0], self.size[1], 0)
			glEnd()


			glfw.swap_buffers(self.window)
			glfw.poll_events()

		core.cuda_sync()



def main(background, use_python_window):
	w, h = 1200, 800

	view = core.View(w,h)

	if not background:
		window = Window(view) if use_python_window else core.Window(view)
		view.clear([0.3, 0.3, 0.3, 1])


	scene = core.Scene()
	createScene(scene)

	pt = core.PathTracer()
	pt.init()
	pt.setScene(scene)
	pt.initLaunchParams(view, 5000)


	pt.render()
	if not background and window.avairable:
		window.draw(pt)

	core.cuda_sync()


	view.downloadImage()

	os.makedirs('result', exist_ok=True)
	plt.imsave('result/out_py.png', view.pixels)


main(background='--background' in sys.argv, use_python_window=True)
main(background='--background' in sys.argv, use_python_window=False)