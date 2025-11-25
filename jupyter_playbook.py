#%%
%load_ext autoreload
%autoreload 2

import pyrender
import numpy as np
import cv2
import trimesh
import matplotlib.pyplot as plt
from cv_utils import (
    load_calibration,
    create_charuco_board,
    estimate_pose_charuco
)

SYSTEM_SCALE = 0.01


def init_scene(scale = 1.0):
    scene = pyrender.Scene()
    mesh = trimesh.load("./3D_models/Jeep_Renegade_2016.obj")
    # Merge all geometries into one mesh
    mesh = sum(mesh.geometry.values()) 
    mesh.apply_scale(scale) 
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    return scene, mesh


#%%
scene, mesh = init_scene(SYSTEM_SCALE)
mesh.bounding_box.extents


#%%
camera_matrix, dist_coeffs = load_calibration("MicrosoftLifeCam_fixedFocus50_calib.npz")
image_width=1280
image_height=720


#%%
frame = np.load("frame.npz")['frame']
board = create_charuco_board(SYSTEM_SCALE)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
T_camera, rvec, tvec = estimate_pose_charuco(
    gray, 
    board, 
    camera_matrix, 
    dist_coeffs
)
cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, board.getSquareLength() * 3)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(frame)

print(T_camera)


#%%
renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)

light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=np.eye(4))

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
camera = pyrender.IntrinsicsCamera(
    fx=camera_matrix[0,0],
    fy=camera_matrix[1,1],
    cx=camera_matrix[0,2],
    cy=camera_matrix[1,2]
)   

camera_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, .01],
    [0, 0, 1, .08],
    [0, 0, 0, 1]
])
camera_node = scene.add(camera, pose=camera_pose)

color, depth = renderer.render(scene)
plt.imshow(depth)


#%%
# Blend the background and rendered image (where color !=  white)
white = np.array([255, 255, 255], dtype=color.dtype)
mask = ~np.all(color == white, axis=-1)  # True where pixel is NOT pure white
mask3 = np.stack([mask]*3, axis=-1)
composite = np.where(mask3, color, frame)
plt.imshow(composite)
print("Mask coverage (percent):", mask.sum() * 100.0 / mask.size)


#%%
# pyrender.Viewer(scene, use_raymond_lighting=True)
