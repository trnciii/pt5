from .scene import Material
from .scene import Scene
from .scene import TriangleMesh
from .scene import Camera

float3_dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
uint3_dtype = [('x', '<i4'), ('y', '<i4'), ('z', '<i4')]

Vertex_dtype = [('p', float3_dtype), ('n', float3_dtype)]
Face_dtype = [
	('vertices', uint3_dtype),
	('uv', uint3_dtype),
	('smooth', 'i1'),
	('material', '<i4')
]

from .view import View
from .tracer import PathTracer

def cuda_sync():pass
