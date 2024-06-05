import numpy as np
from numpy import linalg, dot, cross, sin, cos, sqrt
from numpy.linalg import norm

class quaternion:
    def __init__(self, s, v):
        s = np.array(s)
        v = np.array(v)
        self.s = s
        self.v = v
    
    def __add__(self, q):
        return quaternion(self.s + q.s, self.v + q.v)
    
    def __neg__(self):
        return quaternion(-self.s, -self.v)
    
    def __mul__(self, q):
        s = self.s * q.s - dot(self.v, q.v)
        v = self.s * q.v + q.s * self.v + cross(self.v, q.v)
        return quaternion(s, v)
    
    def __rmul__(self, a):
        return quaternion(a * self.s, a * self.v)
    
    def conjugate(self):
        return quaternion(self.s, -self.v)
    
    def norm(self):
        return pow(self.s**2 + dot(self.v, self.v), 0.5)
    
    def normalize(self):
        return self / self.norm()
    
    def __truediv__(self, q):
        return self * q.conjugate() / (q.norm()**2)
    
    def __repr__(self):
        return "({}, {})".format(self.s, self.v)

def rotation_quaternion(axis, angle):
    axis = axis / norm(axis)
    s = cos(angle / 2)
    v = axis * sin(angle / 2)
    return quaternion(s, v)

def qtndot(q1, q2):
    return q1.s * q2.s + dot(q1.v, q2.v)

def rotate_vector(q, v):
    qv = quaternion(0, v)
    return (q * qv * q.conjugate()).v