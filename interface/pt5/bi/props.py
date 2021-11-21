import bpy
from bpy.props import (
	IntProperty,
	PointerProperty,
)


class PT5Properties(bpy.types.PropertyGroup):

	spp_final: IntProperty(
		default = 1000,
	)

	spp_viewport: IntProperty(
		default = 100,
	)


	@classmethod
	def register(cls):
		bpy.types.Scene.pt5 = PointerProperty(
			name = 'pt5 render settings',
			description = 'pt5 render settings',
			type = cls
		)

	@classmethod
	def unregister(cls):
		del bpy.types.Scene.pt5


def register():
	bpy.utils.register_class(PT5Properties)

def unregister():
	bpy.utils.unregister_class(PT5Properties)