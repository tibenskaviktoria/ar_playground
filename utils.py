import numpy as np
from pytransform3d.rotations import matrix_from_axis_angle

def timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def rotx(theta):
    R = matrix_from_axis_angle((1,0,0, theta))
    T = np.eye(4)
    T[:3,:3] = R
    return T