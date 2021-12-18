from . import core

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders as GLshaders
import ctypes
import numpy as np


class Shader:
	def __init__(self):
		self.shader = None

		src_vert = '''
			#version 330

			layout(location = 0) in vec3 position;
			layout(location = 1) in vec2 uv;

			out vec2 co;

			void main() {
				gl_Position = vec4(position, 1);
				co = uv;
			}
		'''

		src_frag = '''
			#version 330

			in vec2 co;
			out vec4 out_color;

			uniform sampler2D texData;

			void main() {
				out_color = texture(texData, co);
			}
		'''

		self.shader = GLshaders.compileProgram(
			GLshaders.compileShader(src_vert, GL_VERTEX_SHADER),
			GLshaders.compileShader(src_frag, GL_FRAGMENT_SHADER),
		)

		if not self.shader:
			print('failed to create shaders')


	def __del__(self):
		glDeleteProgram(self.shader)


	def use(self):
		glUseProgram(self.shader)

	def loc(self, name):
		return glGetUniformLocation(self.shader, name)


class Polygon:
	def __init__(self):
		coords = np.array([
			-1,-1, 0,  0, 1,
			 1,-1, 0,	 1, 1,
			-1, 1, 0,  0, 0,
			 1, 1, 0,  1, 0,
		], dtype = np.float32)

		self.vertexBuffer = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.vertexBuffer)
		glBufferData(GL_ARRAY_BUFFER, coords, GL_STATIC_DRAW)


		self.vao = glGenVertexArrays(1)
		glBindVertexArray(self.vao)

		glEnableVertexAttribArray(0)
		glVertexAttribPointer(0, 3, GL_FLOAT, False, 20, None)

		glEnableVertexAttribArray(1)
		glVertexAttribPointer(1, 2, GL_FLOAT, False, 20, ctypes.c_void_p(12))


	def __del__(self):
		glDeleteVertexArrays(1, self.vao)
		glDeleteBuffers(1, self.vertexBuffer)


	def draw(self):
		glBindVertexArray(self.vao)
		glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
		# glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0)


class Window:
	def __init__(self, view):
		self.view = view

		if not glfw.init():
			self.window = None
			print('failed to create window')
			return

		glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
		glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
		glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
		self.window = glfw.create_window(view.size[0], view.size[1], 'viewer', None, None)
		if not self.window:
			glfw.terminate()
			return

		glfw.make_context_current(self.window)
		glEnable(GL_FRAMEBUFFER_SRGB);

		glfw.set_key_callback(self.window, self.on_key)


		self.shader = Shader()
		self.rect = Polygon()
		self.view.createGLTexture()


	def __del__(self):
		if not self.window: return
		self.view.destroyGLTexture()
		glfw.destroy_window(self.window)
		glfw.terminate()


	def draw(self, tracer):
		if not self.window:
			core.cuda_sync()
			return

		while (not glfw.window_should_close(self.window))	and tracer.running:
			self.view.updateGLTexture()

			self.shader.use()

			glUniform1i(self.shader.loc('texData'), 0);
			glActiveTexture(GL_TEXTURE0)
			glBindTexture(GL_TEXTURE_2D, self.view.GLTexture)

			self.rect.draw()


			glfw.swap_buffers(self.window)
			glfw.poll_events()

		core.cuda_sync()
		return


	def on_key(self, _win, key, _scancode, action, _mods):
		""" 'Q' or 'Escape' quits """
		if action == glfw.PRESS or action == glfw.REPEAT:
			if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
				glfw.set_window_should_close(self.window, True)
