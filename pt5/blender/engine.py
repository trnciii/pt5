# copied from https://docs.blender.org/api/current/bpy.types.RenderEngine.html

import bpy
import bgl
import gpu
from gpu_extras.batch import batch_for_shader

import pt5
import numpy as np


class CustomRenderEngine(bpy.types.RenderEngine):
	# These three members are used by blender to set up the
	# RenderEngine; define its internal name, visible name and capabilities.
	bl_idname = "PT5"
	bl_label = "pt5"
	bl_use_preview = False
	bl_use_shading_nodes_custom = False
	bl_use_eevee_viewport = True
	bl_use_gpu_context = True

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
		width = int(scene.render.resolution_x * scale)
		height = int(scene.render.resolution_y * scale)

		view = pt5.View(width, height)

		exclude = [o for o in scene.objects if o.hide_render]

		self.tracer.setScene(pt5.scene.create(scene, exclude))
		camera = pt5.scene.camera.fromObject(scene.camera)

		self.tracer.render(view, scene.pt5.spp_final, camera)
		self.tracer.waitForRendering()

		view.downloadImage()
		rect = np.flipud(np.minimum(1, np.maximum(0, view.pixels))).reshape((-1, 4))

		# Here we write the pixel values to the RenderResult
		result = self.begin_result(0, 0, width, height)
		layer = result.layers[0].passes["Combined"]
		layer.rect.foreach_set(rect)
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
		self.tracer.setScene(pt5.scene.create(scene, exclude))


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


		if not self.draw_data or self.draw_data.dimensions != dimensions:
			print('resize')
			self.draw_data = CustomDrawData(dimensions)

		camera = pt5.scene.camera.fromView(context, dimensions)
		self.tracer.render(self.draw_data.view, scene.pt5.spp_viewport, camera)
		self.tracer.waitForRendering()

		self.draw_data.draw()


class CustomDrawData:
	def __init__(self, dimensions):
		self.dimensions = dimensions
		self.view = pt5.View(*dimensions)
		self.view.createGLTexture()

		self.shader = gpu.shader.from_builtin('2D_IMAGE')

		w, h = dimensions
		self.batch = batch_for_shader(self.shader, 'TRI_FAN', {
			'pos':((0,0), (w, 0), (w, h), (0, h)),
			'texCoord': ((0,1), (1,1), (1,0), (0,0))
		})

	def __del__(self):
		self.view.destroyGLTexture()

	def draw(self):
		self.view.updateGLTexture()

		self.shader.bind()
		bgl.glActiveTexture(bgl.GL_TEXTURE0)
		bgl.glBindTexture(bgl.GL_TEXTURE_2D, self.view.GLTexture)
		self.shader.uniform_int('image', 0)
		self.batch.draw(self.shader)


def register():
	# Register the RenderEngine
	bpy.utils.register_class(CustomRenderEngine)


def unregister():
	bpy.utils.unregister_class(CustomRenderEngine)
