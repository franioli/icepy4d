import numpy as np

class DSM:
    def __init__(self, x, y, z, res):
        xx, yy = np.meshgrid(x,y)
        self.x = xx
        self.y = yy
        self.z = z
        self.res = res    