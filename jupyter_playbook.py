#%%
%load_ext autoreload
%autoreload 2

import pyrender
import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt
from pytransform3d.rotations import matrix_from_axis_angle
from pytransform3d.transformations import transform_from as tf
from cv_utils import (
    load_calibration,
    create_charuco_board,
    estimate_pose_charuco
)

SYSTEM_SCALE = 0.05


def init_scene(scale = 1.0):
    scene = pyrender.Scene()
    mesh = trimesh.load("./3D_models/Jeep_Renegade_2016.obj", process=False)
    # Merge all geometries into one mesh
    mesh = sum(mesh.geometry.values()) 
    mesh.apply_scale(scale)
    x_rot = matrix_from_axis_angle((1,0,0, np.pi/2))
    z_rot = matrix_from_axis_angle((0,0,1, -np.pi/2))
    model_rotation = z_rot @ x_rot
    model_corr = tf(p=[0.105, -0.075, 0], R=model_rotation)
    scene.add(pyrender.Mesh.from_trimesh(mesh), pose=model_corr)
    return scene, mesh


def rotx(theta):
    R = matrix_from_axis_angle((1,0,0, theta))
    T = np.eye(4)
    T[:3,:3] = R
    return T


def draw_transformed_axes_on_image(image, camera_matrix, dist_coeffs, A2B, length=0.05):
    tvec = A2B[:3, 3]
    rvec = cv2.Rodrigues(A2B[:3, :3])[0]
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length)
    return image


#%%
scene, mesh = init_scene(SYSTEM_SCALE)
mesh.bounding_box.extents
print(mesh.extents)


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

frame = draw_transformed_axes_on_image(
    frame, 
    camera_matrix, 
    dist_coeffs,
    T_camera_corrected, 
    length=board.getSquareLength() * 3
)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)
print("T_camera", T_camera)


#%%

light = pyrender.DirectionalLight(color=np.ones(3), intensity=50.0)
scene.add(light, pose=np.eye(4))

camera = pyrender.IntrinsicsCamera(
    fx=camera_matrix[0,0],
    fy=camera_matrix[1,1],
    cx=camera_matrix[0,2],
    cy=camera_matrix[1,2]
)

# Assign the camera pose
corr = tf(p=[0,0,0], R=matrix_from_axis_angle((1,0,0, np.pi)))  # correct for OpenCV to OpenGL coord sys
camera_node = scene.add(camera, pose=T_cam2world @ corr)

renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)
color, depth = renderer.render(scene)
plt.imshow(color)


#%%
# Blend the background and rendered image (where color !=  white)
# FIXME: Improve the mask to avoid white parts inside the model
white = np.array([255, 255, 255], dtype=color.dtype)
mask = ~np.all(color == white, axis=-1)  # True where pixel is NOT pure white
mask3 = np.stack([mask]*3, axis=-1)
composite = np.where(mask3, color, frame)
plt.imshow(composite)
print("Mask coverage (percent):", mask.sum() * 100.0 / mask.size)


#%%
#%matplotlib tk
%matplotlib widget

# Visualize the camera position and rotation in world space
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as pu


scene, mesh = init_scene(SYSTEM_SCALE)

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

bounds = scene.bounds
extents = bounds[1] - bounds[0]
center = (bounds[0] + bounds[1]) / 2
A2B = np.eye(4)
A2B[:3, 3] = center
pu.plot_box(ax, extents, A2B=A2B, color="magenta", alpha=0.5)


#%%
