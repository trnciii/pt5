from .core import *
from .window import Window as Window_py


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
	from .bi.engine import register, unregister

del importlib
