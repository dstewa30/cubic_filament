from modules import quaternions
import numpy as np

rot_ax = np.array([0,0,1])
theta = np.pi/4
p = np.array([1,1,1])
p_prime = np.array([-1,-1,1])

v_rot = quaternions.rotation_quaternion(rot_ax,theta)
print(v_rot)
p_prime = quaternions.rotate_vector(v_rot,p)

print(p_prime)
