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


def init_scene(scale = 1.0):
    scene = pyrender.Scene()
    mesh = trimesh.load("./3D_models/Jeep_Renegade_2016.obj", process=False)
    # Merge all geometries into one mesh
    mesh = sum(mesh.geometry.values()) 
    mesh.apply_scale(scale)
    flip_z = np.eye(4)
    flip_z[2, 2] = -1
    model_pose = np.eye(4)
    # model_pose[:3, 3] = [0.1, 0.2, -0.4]
    model_pose[:3, 3] = [0, 0, 0]
    scene.add(pyrender.Mesh.from_trimesh(mesh), pose=model_pose)
    return scene, mesh, model_pose


def rotx(theta):
    R = matrix_from_axis_angle((1,0,0, theta))
    T = np.eye(4)
    T[0:3,0:3] = R
    return T


#%%
scene, mesh, model_pose = init_scene(SYSTEM_SCALE)
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
# T_cam2world = T_camera_corrected

# cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, board.getSquareLength() * 3)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)
print("T_camera", T_camera)


#%%
renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)

light = pyrender.DirectionalLight(color=np.ones(3), intensity=50.0)
scene.add(light, pose=np.eye(4))

cube = trimesh.creation.box(extents=[0.03, 0.03, 0.03])  # 3 cm cube
cube_pyr = pyrender.Mesh.from_trimesh(cube)
scene.add(cube_pyr, pose=np.eye(4))

camera = pyrender.IntrinsicsCamera(
    fx=camera_matrix[0,0],
    fy=camera_matrix[1,1],
    cx=camera_matrix[0,2],
    cy=camera_matrix[1,2]
)

test_cam_pose = np.eye(4)
test_cam_pose = np.array([
    [1, 0, 0, 0.2],
    [0, 1, 0, 0.4],
    [0, 0, 1, -0.5],
    [0, 0, 0, 1]
])

combined_pose = test_cam_pose @ T_cam2world
print("Combined pose:\n", combined_pose)

# Assign the camera pose
selected_cam_pose = combined_pose
camera_node = scene.add(camera, pose=selected_cam_pose)

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

# Visualize the camera position and rotation in world space
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as pu

virtual_image_distance = .5

plt.figure(num='3d', clear=True)
ax = pt.plot_transform(A2B=np.eye(4), s=0.2, name = 'world')  # world origin axes
pt.plot_transform(A2B=selected_cam_pose, ax=ax, s=0.2, name = 'camera')
pc.plot_camera(
    ax,
    cam2world=selected_cam_pose,
    M=camera_matrix,
    sensor_size=(1280,720),
    virtual_image_distance=virtual_image_distance,
)

bounds_min, bounds_max = mesh.bounds
center_model = (bounds_min + bounds_max) / 2.0
T_center = np.eye(4)
T_center[:3, 3] = center_model
pu.plot_box(ax, mesh.extents, A2B=model_pose @ T_center, color="cyan", alpha=0.5)

plt.show()


#%%
