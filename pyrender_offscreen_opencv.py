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
    openCapture
)

def init_scene(scale = 1.0):
    scene = pyrender.Scene()
    mesh = trimesh.load("./3D_models/Jeep_Renegade_2016.obj")
    # Merge all geometries into one mesh
    mesh = sum(mesh.geometry.values()) 
    mesh.apply_scale(scale) 
    scene.add(pyrender.Mesh.from_trimesh(mesh))
    return scene, mesh

SYSTEM_SCALE = 0.01

#%%

scene, mesh = init_scene(SYSTEM_SCALE)

mesh.bounding_box.extents

#%%

# Load calibration
camera_matrix, dist_coeffs = load_calibration("MicrosoftLifeCam_fixedFocus50_calib.npz")
image_width=1280
image_height=720

##%%

renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)

light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=np.eye(4))  # Add light at the default pose

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
    [0, 0, 1, .08],  # Move the camera 4 units away from the origin
    [0, 0, 0, 1]
])
camera_node = scene.add(camera, pose=camera_pose)

color, depth = renderer.render(scene)
plt.imshow(depth, cmap='jet')

#%%

frame = np.load("frame.npz")['frame']
plt.imshow(frame)

#%%

from cv_utils import create_charuco_board, estimate_pose_charuco

board = create_charuco_board(SYSTEM_SCALE)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
T_camera, rvec, tvec = estimate_pose_charuco(
    gray, 
    board, 
    camera_matrix, 
    dist_coeffs
)
cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, board.getSquareLength() * 3)
plt.imshow(frame)

print(T_camera)

##%%

camera_node.matrix = T_camera

##%%

#pyrender.Viewer(scene, use_raymond_lighting=True)

#%%

color, depth = renderer.render(scene)

plt.imshow(color)

#%%

axis = trimesh.creation.axis(origin_size=.5)
mesh = trimesh.load("./3D_models/Jeep_Renegade_2016.obj", process=True)

trimesh_scene = axis.scene()
trimesh_scene.add_geometry(mesh)
trimesh_scene.camera.transform = T_camera
trimesh_scene.show()

#%%

#%%


frame = None
with openCapture(0, image_width, image_height) as cap:
    ret, frame = cap.read()

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)

np.savez("frame.npz", frame=frame)

