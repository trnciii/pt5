import bpy

class PT5_PT_sampling(bpy.types.Panel):
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


def register():
	bpy.utils.register_class(PT5_PT_sampling)


def unregister():
	bpy.utils.unregister_class(PT5_PT_sampling)
