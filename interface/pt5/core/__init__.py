import importlib

if importlib.util.find_spec(".core", package='pt5.core'):
	from .core import *
else:
	print('pt5 using dummy core')
	from .core_dummy import *

del importlib
