from .core import *

# setup to work with blender

bl_info = {
  "name": "pt5",
  "blender": (2, 93, 0),
  "category": "Render",
}

import importlib
if importlib.util.find_spec("bpy"):
	print('import blender interface')
	from .bi import *

	def register():
		from .bi import engine, panels, props

		engine.register()
		panels.register()
		props.register()


	def unregister():
		from .bi import engine, panels, props

		engine.unregister()
		panels.unregister()
		props.unregister()


del importlib
