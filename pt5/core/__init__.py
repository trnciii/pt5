from importlib import util

if util.find_spec(".core", package='pt5.core'):
	from .core import *
else:
	print('pt5 using dummy core')
	from .dummy import *

del util
