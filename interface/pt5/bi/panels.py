import bpy
from bpy_extras.node_utils import find_node_input
from cycles.ui import *



class PT5_Panel_Base:
	COMPAT_ENGINES = {'PT5'}

	@classmethod
	def poll(cls, context):
		return context.engine in cls.COMPAT_ENGINES



class PT5_PT_sampling(PT5_Panel_Base, bpy.types.Panel):
	bl_label = "Sampling"
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "render"

	def draw(self, context):
		pt5 = context.scene.pt5
		layout = self.layout

		layout.use_property_split = True
		layout.use_property_decorate = False

		col = layout.column(align = True)
		col.prop(pt5, 'spp_final', text = 'Final')
		col.prop(pt5, 'spp_viewport', text = 'Viewport')



class PT5_PT_context_material(PT5_Panel_Base, bpy.types.Panel):
	bl_label = ''
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "material"
	bl_options = {'HIDE_HEADER'}

	draw = CYCLES_PT_context_material.draw

	@classmethod
	def poll(cls, context):
		if context.active_object and context.active_object.type == 'GPENCIL':
			return False
		else:
			return (context.material or context.object) and PT5_Panel_Base.poll(context)



class PT5_MATERIAL_PT_surface(bpy.types.Panel):
	bl_label = 'Surface'
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = 'material'

	draw = CYCLES_MATERIAL_PT_surface.draw

	@classmethod
	def poll(cls, context):
		m = context.material
		return m and (not m.grease_pencil) and PT5_Panel_Base.poll(context)



class PT5_WORLD_PT_surface(PT5_Panel_Base, bpy.types.Panel):
	bl_label = "Surface"
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "world"

	draw = CYCLES_WORLD_PT_surface.draw

	@classmethod
	def poll(cls, context):
		return context.world and PT5_Panel_Base.poll(context)



classes = (
	PT5_PT_sampling,
	PT5_PT_context_material,
	PT5_MATERIAL_PT_surface,
	PT5_WORLD_PT_surface,
)


# RenderEngines also need to tell UI Panels that they are compatible with.
# We recommend to enable all panels marked as BLENDER_RENDER, and then
# exclude any panels that are replaced by custom panels registered by the
# render engine, or that are not supported.
def get_panels():
	exclude_panels = {
		'VIEWLAYER_PT_filter',
		'VIEWLAYER_PT_layer_passes',
	}

	panels = []
	for panel in bpy.types.Panel.__subclasses__():
		if hasattr(panel, 'COMPAT_ENGINES') and 'BLENDER_RENDER' in panel.COMPAT_ENGINES:
			if panel.__name__ not in exclude_panels:
				panels.append(panel)

	return panels


def register():
	for c in classes:
		bpy.utils.register_class(c)

	for panel in get_panels():
		panel.COMPAT_ENGINES.add('PT5')


def unregister():
	for c in classes:
		bpy.utils.unregister_class(c)

	for panel in get_panels():
		if 'PT5' in panel.COMPAT_ENGINES:
			panel.COMPAT_ENGINES.remove('PT5')
