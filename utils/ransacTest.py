import pickle
import pptk


xyz = pptk.rand(100, 3)
v = pptk.viewer(xyz)
v.set(point_size=0.005)