from . import core

import glfw
from OpenGL.GL import *


class Window:
	def __init__(self, view):
		self.view = view


		if not glfw.init():
			self.window = None
			return

		self.size = view.size
		self.window = glfw.create_window(self.size[0], self.size[1], "python window", None, None)
		if not self.window:
			glfw.terminate()
			return


		glfw.make_context_current(self.window)

		glEnable(GL_FRAMEBUFFER_SRGB)
		glViewport(0,0,self.size[0], self.size[1])
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		glOrtho(0, self.size[0], 0, self.size[1], -1, 1)

		self.view.createGLTexture()


	def __del__(self):
		print('WINDOW DESTRUCTOR')
		if not self.window: return
		self.view.destroyGLTexture()
		glfw.destroy_window(self.window)
		glfw.terminate()


	@property
	def hasWindow(self):
		return self.window != None


	def draw(self, pt):
		if not self.window:
			core.cuda_sync()
			return

		glBindTexture(GL_TEXTURE_2D, self.view.GLTexture)


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