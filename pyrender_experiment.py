#%%
%load_ext autoreload
%autoreload 2

import pyrender
import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt
from pytransform3d.rotations import matrix_from_axis_angle
from cv_utils import (
    load_calibration,
    create_charuco_board,
    estimate_pose_charuco
)

SYSTEM_SCALE = 0.05


def init_scene(scale = 1.0, rot_x = 0.0, rot_y = 0.0, rot_z = 0.0):
    scene = pyrender.Scene()
    mesh = trimesh.load("./3D_models/Jeep_Renegade_2016.obj", process=False)
    # Merge all geometries into one mesh
    mesh = sum(mesh.geometry.values()) 
    mesh.apply_scale(scale)

    scene.add(pyrender.Mesh.from_trimesh(mesh))
    return scene, mesh


def rotx(theta):
    R = matrix_from_axis_angle((1,0,0, theta))
    T = np.eye(4)
    T[0:3,0:3] = R
    return T


def drawTransformOnImage(image, camera_matrix, dist_coeffs, A2B, length=0.05):
    tvec = A2B[:3, 3]
    rvec = cv2.Rodrigues(A2B[:3, :3])[0]
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length)
    return image


#%%

scene, mesh = init_scene(SYSTEM_SCALE)

print("Scene bounds: \n", scene.bounds)
print("Mesh bounding box: \n", mesh.bounding_box.extents)


#%%

camera_matrix, dist_coeffs = load_calibration("MicrosoftLifeCam_fixedFocus50_calib.npz")
image_width=1280
image_height=720


#%%

frame = np.load("frame.npz")['frame']
board = create_charuco_board()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
T_camera, rvec, tvec = estimate_pose_charuco(
    gray, 
    board, 
    camera_matrix, 
    dist_coeffs
)

T_camera_corrected = T_camera @ rotx(np.pi)
T_cam2world = np.linalg.inv(T_camera_corrected)

frame = drawTransformOnImage(frame, camera_matrix, dist_coeffs, 
                             T_camera_corrected, 
                             length=board.getSquareLength() * 3)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)
print("T_cam2world: \n", T_cam2world)

#%%

%matplotlib widget

# Visualize the camera position and rotation in world space
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as pu


scene, mesh = init_scene(.05)

virtual_image_distance = .35

plt.figure(figsize=(5,5))
ax = pt.plot_transform(ax_s=.35, A2B=np.eye(4), s=0.2, name = 'world')  # world origin axes
ax.view_init(elev=40, azim=-160,roll=0)
pt.plot_transform(A2B=T_cam2world, ax=ax, s=0.2, name = 'camera')
pc.plot_camera(
    ax,
    cam2world=T_cam2world,
    M=camera_matrix,
    sensor_size=(1280,720),
    virtual_image_distance=virtual_image_distance,
)

bb = mesh.bounding_box
pu.plot_box(ax, bb.extents, A2B=bb.transform, color="cyan", alpha=0.5)

#%%

import trimesh
from trimesh.creation import axis

origin = axis(origin_size=0.1, axis_length=1)
scene = origin.scene()
mesh1 = trimesh.load("./3D_models/Jeep_Renegade_2016.obj", process=False)
scene.add_geometry(mesh1)
scene.show()

#%%

from pytransform3d.transformations import transform_from as tf
from pytransform3d.rotations import matrix_from_axis_angle

def rotx(theta):
    return matrix_from_axis_angle((1,0,0, theta))

def roty(theta):
    return matrix_from_axis_angle((0,1,0, theta))

def rotz(theta):
    return matrix_from_axis_angle((0,0,1, theta))

camera = pyrender.IntrinsicsCamera(
    fx = camera_matrix[0,0],
    fy = camera_matrix[1,1],
    cx = camera_matrix[0,2],
    cy = camera_matrix[1,2]
)
scene, mesh = init_scene(SYSTEM_SCALE)
print(mesh.bounding_box.extents)

corr = tf(p=[0,0,0], R=rotx(np.pi))  # correct for OpenCV to OpenGL coord sys
scene.add(camera, pose = T_cam2world @ corr)

renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)
color, depth = renderer.render(scene)
plt.figure()
plt.imshow(color)



#%%
