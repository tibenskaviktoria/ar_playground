"""
AR Playground - Pyrender Offscreen with OpenCV

This module demonstrates augmented reality (AR) by rendering a 3D model on top of a live camera feed.
It uses a ChArUco board for camera pose estimation and pyrender for offscreen 3D rendering, compositing
the rendered model onto the video stream based on the detected board pose.
"""
import numpy as np
import cv2
import pyrender
import trimesh
from contextlib import ExitStack
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import matrix_from_axis_angle
from cv_utils import (
    load_calibration,
    openCapture,
    createWindow,
    create_charuco_board,
    estimate_pose_charuco,
    draw_transformed_axes_on_image
)
from utils import rotx


def estimate_camera_pose(board, frame, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    T_camera = estimate_pose_charuco(
        gray, 
        board, 
        camera_matrix, 
        dist_coeffs
    )
    if T_camera is None:
        return None, frame
    
    T_camera_corrected = T_camera @ rotx(np.pi)
    T_cam2world = np.linalg.inv(T_camera_corrected)

    frame = draw_transformed_axes_on_image(
        frame, 
        camera_matrix, 
        dist_coeffs,
        T_camera_corrected, 
        length=board.getSquareLength() * 3
    )
    return T_cam2world, frame


def load_and_add_model_to_scene(pyr_scene, model_path, p = [0, 0, 0], rotation = [0, 0, 0], model_scale = 1.0):
    """
    Initialize a pyrender scene with a 3D model.
    Loads a 3D model from file, applies transformations, and adds it to the pyrender scene.
    The model is scaled, rotated, and positioned according to the specified parameters.
    Parameters:
        pyr_scene: The pyrender scene object to which the model will be added.
        model_path (str): Path to the 3D model file to be loaded.
        p (list, optional): Position [x, y, z] of the model in 3D space. Defaults to [0, 0, 0].
        rotation (list, optional): Rotation angles [x, y, z] for the model. Defaults to [0, 0, 0].
        model_scale (float, optional): Scale factor to apply to the model. Defaults to 1.0.
    Returns:
        trimesh.Trimesh: The merged and transformed mesh object.
    """
    mesh = trimesh.load(model_path, process=False)
    mesh = sum(mesh.geometry.values()) # Merge all geometries into one mesh
    mesh.apply_scale(model_scale)
    x_rot = matrix_from_axis_angle((1,0,0, rotation[0]))
    y_rot = matrix_from_axis_angle((0,1,0, rotation[1]))
    z_rot = matrix_from_axis_angle((0,0,1, rotation[2]))
    model_rotation = z_rot @ x_rot
    model_correction = transform_from(p=p, R=model_rotation)
    pyr_scene.add(pyrender.Mesh.from_trimesh(mesh), pose=model_correction)
    return mesh


def add_camera_to_scene(pyr_scene, camera_matrix, T_cam2world):
    if T_cam2world is None:
        return False
    
    camera = pyrender.IntrinsicsCamera(
        fx=camera_matrix[0,0],
        fy=camera_matrix[1,1],
        cx=camera_matrix[0,2],
        cy=camera_matrix[1,2]
    )

    corr = transform_from(p=[0,0,0], R=matrix_from_axis_angle((1,0,0, np.pi)))  # correct for OpenCV to OpenGL coord sys
    pyr_scene.add(camera, pose=T_cam2world @ corr)
    return True


def create_depth_mask(depth, color, frame):
    """Create a binary mask where depth is valid (greater than zero)."""
    mask = depth > 0
    mask3 = np.stack([mask]*3, axis=-1)
    composite = np.where(mask3, color, frame)
    return composite


def main():
    camera_matrix, dist_coeffs = load_calibration("./calibration/MicrosoftLifeCam_fixedFocus50_calib.npz")
    if camera_matrix is None:
        print("No calibration found. Exiting...")
        return

    image_width=1280
    image_height=720
    board = create_charuco_board()
    scene = pyrender.Scene()

    with ExitStack() as stack:
        renderer = pyrender.OffscreenRenderer(viewport_width=image_width, viewport_height=image_height)
        stack.callback(renderer.delete)
        
        stack.enter_context(createWindow("AR Camera"))
        # FIXME: the camera seems to work only when the script is ran for the first time
        cap = stack.enter_context(openCapture(0))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            T_cam2world, frame = estimate_camera_pose(board, frame, camera_matrix, dist_coeffs)

            load_and_add_model_to_scene(
                scene,
                "./3D_models/Jeep_Renegade_2016.obj",
                model_scale=0.05,
                rotation = [np.pi/2, 0, -np.pi/2],
                p = [0.105, -0.075, 0]
            )
            
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=20.0)
            scene.add(light, pose=np.eye(4))

            camera_added_succ = add_camera_to_scene(scene, camera_matrix, T_cam2world)

            if camera_added_succ:
                color, depth = renderer.render(scene)
                composite_img = create_depth_mask(depth, color, frame)
                cv2.imshow("AR Camera", composite_img)
            else:
                cv2.imshow("AR Camera", frame)
            
            scene.clear()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): # Quit
                break


if __name__ == "__main__":
    main()
