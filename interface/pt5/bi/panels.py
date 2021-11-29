import bpy
from bpy_extras.node_utils import find_node_input


def panel_node_draw(layout, id_data, output_type, input_name):
	if not id_data.use_nodes:
		layout.operator("cycles.use_shading_nodes", icon='NODETREE')
		return False

	ntree = id_data.node_tree

	node = ntree.get_output_node('CYCLES')
	if node:
		input = find_node_input(node, input_name)
		if input:
		    layout.template_node_view(ntree, node, input)
		else:
		    layout.label(text="Incompatible output node")
	else:
		layout.label(text="No output node")

	return True



class PT5_Panel_Base:
	@classmethod
	def poll(cls, context):
		return context.engine == 'PT5'



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


class PT5_PT_materials(PT5_Panel_Base, bpy.types.Panel):
	bl_label = ""
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "material"
	bl_options = {'HIDE_HEADER'}

	@classmethod
	def poll(cls, context):
		if context.active_object and context.active_object.type == 'GPENCIL':
			return False
		else:
			return (context.material or context.object) and PT5_Panel_Base.poll(context)

	def draw(self, context):
		layout = self.layout

		mat = context.material
		ob = context.object
		slot = context.material_slot
		space = context.space_data

		if ob:
			is_sortable = len(ob.material_slots) > 1
			rows = 1
			if (is_sortable):
				rows = 4

			row = layout.row()

			row.template_list("MATERIAL_UL_matslots", "", ob, "material_slots", ob, "active_material_index", rows=rows)

			col = row.column(align=True)
			col.operator("object.material_slot_add", icon='ADD', text="")
			col.operator("object.material_slot_remove", icon='REMOVE', text="")

			col.menu("MATERIAL_MT_context_menu", icon='DOWNARROW_HLT', text="")

			if is_sortable:
				col.separator()

				col.operator("object.material_slot_move", icon='TRIA_UP', text="").direction = 'UP'
				col.operator("object.material_slot_move", icon='TRIA_DOWN', text="").direction = 'DOWN'

			if ob.mode == 'EDIT':
				row = layout.row(align=True)
				row.operator("object.material_slot_assign", text="Assign")
				row.operator("object.material_slot_select", text="Select")
				row.operator("object.material_slot_deselect", text="Deselect")

		split = layout.split(factor=0.65)

		if ob:
			split.template_ID(ob, "active_material", new="material.new")
			row = split.row()

			if slot:
				row.prop(slot, "link", text="")
			else:
				row.label()
		elif mat:
			split.template_ID(space, "pin_id")
			split.separator()


class PT5_PT_material_surface(PT5_Panel_Base, bpy.types.Panel):
	bl_label = 'Surface'
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = 'material'

	@classmethod
	def poll(cls, context):
		m = context.material
		return m and (not m.grease_pencil) and PT5_Panel_Base.poll(context)


	def draw(self, context):
		layout = self.layout
		layout.use_property_split = True
		mat = context.material
		if not panel_node_draw(layout, mat, 'OUTPUT_MATERIAL', 'Surface'):
			layout.prop(mat, 'diffuse_color')


class PT5_PT_world(PT5_Panel_Base, bpy.types.Panel):
	bl_label = "Surface"
	bl_space_type = 'PROPERTIES'
	bl_region_type = 'WINDOW'
	bl_context = "world"


	@classmethod
	def poll(cls, context):
		return context.world and PT5_Panel_Base.poll(context)


	def draw(self, context):
		layout = self.layout
		layout.use_property_split = True
		world = context.world
		if not panel_node_draw(layout, world, 'OUTPUT_WORLD', 'Surface'):
			layout.prop(world, "color")



classes = (
	PT5_PT_sampling,
	PT5_PT_materials,
	PT5_PT_material_surface,
	PT5_PT_world,
)

def register():
	for c in classes:
		bpy.utils.register_class(c)


def unregister():
	for c in classes:
		bpy.utils.unregister_class(c)
