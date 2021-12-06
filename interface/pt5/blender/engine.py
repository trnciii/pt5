# copied from https://docs.blender.org/api/current/bpy.types.RenderEngine.html

import bpy
import bgl

import pt5
import numpy as np


class CustomRenderEngine(bpy.types.RenderEngine):
	# These three members are used by blender to set up the
	# RenderEngine; define its internal name, visible name and capabilities.
	bl_idname = "PT5"
	bl_label = "pt5"
	bl_use_preview = True
	bl_use_shading_nodes_custom = False

	# Init is called whenever a new render engine instance is created. Multiple
	# instances may exist at the same time, for example for a viewport and final
	# render.
	def __init__(self):
		print()
		print('engine init')

		self.scene_data = None
		self.draw_data = None

		self.tracer = pt5.PathTracer()


	# When the render engine instance is destroy, this is called. Clean up any
	# render engine data here, for example stopping running render threads.
	def __del__(self):
		print('engine delete')
		print()

	# This is the method called by Blender for both final renders (F12) and
	# small preview for materials, world and lights.
	def render(self, depsgraph):
		print('final render')

		scene = depsgraph.scene
		scale = scene.render.resolution_percentage / 100.0
		self.size_x = int(scene.render.resolution_x * scale)
		self.size_y = int(scene.render.resolution_y * scale)

		view = pt5.View(self.size_x, self.size_y)

		exclude = [o for o in scene.objects if o.hide_render]

		self.tracer.setScene(pt5.scene.createSceneFromBlender(scene, exclude))
		camera = pt5.scene.createCameraFromObject(scene.camera)

		self.tracer.render(view, scene.pt5.spp_final, camera)
		pt5.cuda_sync()

		view.downloadImage()
		rect = np.flipud(np.minimum(1, np.maximum(0, view.pixels))).reshape((-1, 4))

		# Here we write the pixel values to the RenderResult
		result = self.begin_result(0, 0, self.size_x, self.size_y)
		layer = result.layers[0].passes["Combined"]
		layer.rect = rect
		self.end_result(result)

	# For viewport renders, this method gets called once at the start and
	# whenever the scene or 3D viewport changes. This method is where data
	# should be read from Blender in the same thread. Typically a render
	# thread will be started to do the work while keeping Blender responsive.
	def view_update(self, context, depsgraph):
		region = context.region
		view3d = context.space_data
		scene = depsgraph.scene

		exclude = [o for o in scene.objects if o.hide_get()]

		self.tracer.removeScene()
		self.tracer.setScene(pt5.scene.createSceneFromBlender(scene, exclude))


		if not self.scene_data:
			# First time initialization
			self.scene_data = []
			first_time = True

			# Loop over all datablocks used in the scene.
			for datablock in depsgraph.ids:
				pass
		else:
			first_time = False

			# Test which datablocks changed
			for update in depsgraph.updates:
				print("Datablock updated: ", update.id.name)

			# Test if any material was added, removed or changed.
			if depsgraph.id_type_updated('MATERIAL'):
				print("Materials updated")

		# Loop over all object instances in the scene.
		if first_time or depsgraph.id_type_updated('OBJECT'):
			for instance in depsgraph.object_instances:
				pass

	# For viewport renders, this method is called whenever Blender redraws
	# the 3D viewport. The renderer is expected to quickly draw the render
	# with OpenGL, and not perform other expensive work.
	# Blender will draw overlays for selection and editing on top of the
	# rendered image automatically.
	def view_draw(self, context, depsgraph):
		region = context.region
		space = context.space_data
		scene = depsgraph.scene

		# Get viewport dimensions
		dimensions = region.width, region.height
		if space.use_render_border:
			border = (space.render_border_min_x,
				space.render_border_min_y,
				space.render_border_max_x,
				space.render_border_max_y)
		else:
			border = (0, 0, 1, 1)


		# Bind shader that converts from scene linear to display space,
		bgl.glEnable(bgl.GL_BLEND)
		bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
		self.bind_display_space_shader(scene)

		if not self.draw_data or self.draw_data.dimensions != dimensions:
			print('resize')
			self.view = pt5.View(*dimensions)
			self.draw_data = CustomDrawData(dimensions, self.view)


		self.camera = pt5.scene.getViewAsCamera(context, dimensions)

		self.tracer.render(self.view, scene.pt5.spp_viewport, self.camera)
		pt5.cuda_sync()

		self.draw_data.draw()

		self.unbind_display_space_shader()
		bgl.glDisable(bgl.GL_BLEND)


class CustomDrawData:
	def __init__(self, dimensions, view):
		# Generate dummy float image buffer
		self.dimensions = dimensions
		width, height = dimensions

		self.view = view
		self.view.createGLTexture()

		# Bind shader that converts from scene linear to display space,
		# use the scene's color management settings.
		shader_program = bgl.Buffer(bgl.GL_INT, 1)
		bgl.glGetIntegerv(bgl.GL_CURRENT_PROGRAM, shader_program)

		# Generate vertex array
		self.vertex_array = bgl.Buffer(bgl.GL_INT, 1)
		bgl.glGenVertexArrays(1, self.vertex_array)
		bgl.glBindVertexArray(self.vertex_array[0])

		texturecoord_location = bgl.glGetAttribLocation(shader_program[0], "texCoord")
		position_location = bgl.glGetAttribLocation(shader_program[0], "pos")

		bgl.glEnableVertexAttribArray(texturecoord_location)
		bgl.glEnableVertexAttribArray(position_location)

		# Generate geometry buffers for drawing textured quad
		position = [0.0, 0.0, width, 0.0, width, height, 0.0, height]
		position = bgl.Buffer(bgl.GL_FLOAT, len(position), position)
		texcoord = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
		texcoord = bgl.Buffer(bgl.GL_FLOAT, len(texcoord), texcoord)

		self.vertex_buffer = bgl.Buffer(bgl.GL_INT, 2)

		bgl.glGenBuffers(2, self.vertex_buffer)
		bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[0])
		bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, position, bgl.GL_STATIC_DRAW)
		bgl.glVertexAttribPointer(position_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

		bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, self.vertex_buffer[1])
		bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, texcoord, bgl.GL_STATIC_DRAW)
		bgl.glVertexAttribPointer(texturecoord_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)

		bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)
		bgl.glBindVertexArray(0)

	def __del__(self):
		bgl.glDeleteBuffers(2, self.vertex_buffer)
		bgl.glDeleteVertexArrays(1, self.vertex_array)
		self.view.destroyGLTexture()

	def draw(self):
		self.view.updateGLTexture()

		bgl.glActiveTexture(bgl.GL_TEXTURE0)
		bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.view.GLTexture)
		bgl.glBindVertexArray(self.vertex_array[0])
		bgl.glDrawArrays(bgl.GL_TRIANGLE_FAN, 0, 4)
		bgl.glBindVertexArray(0)
		bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)



def register():
	# Register the RenderEngine
	bpy.utils.register_class(CustomRenderEngine)


def unregister():
	bpy.utils.unregister_class(CustomRenderEngine)
