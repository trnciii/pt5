from pt5 import BScene, core
import numpy as np
import matplotlib.pyplot as plt
import bpy


scale = bpy.context.scene.render.resolution_percentage/100
width = int(bpy.context.scene.render.resolution_x*scale)
height = int(bpy.context.scene.render.resolution_y*scale)


scene = core.Scene()
BScene.createScene(scene)


pt = core.PathTracer()
pt.init()
pt.setScene(scene)
pt.initLaunchParams(width, height, 100)

pt.render()

pixels4 = np.array(np.minimum(1, np.maximum(0, pt.pixels()**0.4))).reshape((height, width, 4))
plt.imsave('out_blender.png', pixels4)