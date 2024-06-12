import numpy as np
from cylindermath import norm

class block:
    def __init__(self, xlo, xhi, ylo, yhi, zlo, zhi):
        self.xlo = xlo
        self.xhi = xhi
        self.ylo = ylo
        self.yhi = yhi
        self.zlo = zlo
        self.zhi = zhi
        self.w_x = xhi - xlo
        self.w_y = yhi - ylo
        self.w_z = zhi - zlo
        
def distance_from_axis(block, rP):
    px, py, pz = rP[0], rP[1], rP[2]
    
    d = np.array([0, py, pz])
    
    return norm(d)

def distance_from_surface(block, rP):
    px, py, pz = rP[0], rP[1], rP[2]
    
    dy1 = py
    dy2 = block.w_y - py
    dz1 = pz
    dz2 = block.w_z - pz
    d1x = px
    d2x = block.w_x - px
    
    dx = min(d1x, d2x)
    dy = min(dy1, dy2)
    dz = min(dz1, dz2)
    
    return dx, dy, dz