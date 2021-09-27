import core

import matplotlib.pyplot as plt
import numpy as np

w, h = 1200, 800

scene = core.Scene()
scene.createDefault()

pt = core.PathTracer()
pt.init()
pt.setScene(scene)
pt.initLaunchParams(w, h)

pt.render()

pixels4 = np.array(pt.pixels()).reshape((h,w,4))

plt.imshow(pixels4)
plt.show()

plt.imsave('out_py.png', pixels4)