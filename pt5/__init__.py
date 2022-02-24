from .core import *
from . import dtype

setNodeIndices()

# setup to work with blender

bl_info = {
  "name": "pt5",
  "blender": (2, 93, 0),
  "category": "Render",
}

from importlib import util
if util.find_spec("bpy"):
	print('import pt5.blender module')
	from .blender import scene

	def register():
		from .blender import engine, panels, props

		engine.register()
		panels.register()
		props.register()


	def unregister():
		from .blender import engine, panels, props

		engine.unregister()
		panels.unregister()
		props.unregister()

del util
