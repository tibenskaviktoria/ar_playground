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
    model_pose[:3, 3] = [0, 0, 0.02]
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    return scene, mesh


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
cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, board.getSquareLength() * 3)
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

test_cam_pose = np.eye(4)
test_cam_pose = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0.4],
    [0, 0, 0, 1]
])

combined_pose = test_cam_pose @ T_camera
print("Combined pose:\n", combined_pose)

camera = pyrender.IntrinsicsCamera(
    fx=camera_matrix[0,0],
    fy=camera_matrix[1,1],
    cx=camera_matrix[0,2],
    cy=camera_matrix[1,2]
)   

# Assign the camera pose
selected_cam_pose = test_cam_pose
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
# Visualize the camera position and rotation in world space
import pytransform3d.camera as pc
import pytransform3d.transformations as pt

cam2world = pt.transform_from_pq([0, 0, 0, np.sqrt(0.5), -np.sqrt(0.5), 0, 0])
sensor_size = np.array([0.036, 0.024]) # default parameters of a camera in Blender
intrinsic_matrix = np.array(
    [
        [0.05, 0, sensor_size[0] / 2.0],
        [0, 0.05, sensor_size[1] / 2.0],
        [0, 0, 1],
    ]
)
virtual_image_distance = 1

ax = pt.plot_transform(A2B=np.eye(4), s=0.2)  # world origin axes
pt.plot_transform(A2B=selected_cam_pose, ax=ax, s=0.2)
pc.plot_camera(
    ax,
    cam2world=selected_cam_pose,
    M=intrinsic_matrix,
    sensor_size=sensor_size,
    virtual_image_distance=virtual_image_distance,
)
plt.show()


#%%
