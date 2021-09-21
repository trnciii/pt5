import core

w, h = 1200, 800

pt = core.PathTracer()
pt.buildSBT()
pt.initLaunchParams(w, h)

pt.render()

core.writeImage('out_py.png', w, h, pt.pixels())