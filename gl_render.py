# Create a program that uses the view from the camera and draws a 3D cube on a detected ArUco marker.
from contextlib import ExitStack
import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from cv_utils import (
    load_calibration,
    create_charuco_board,
    openCapture
)
from gl_utils import (
    draw_cube_gl,
    setup_projection,
    calculateGlProjection,
    draw_axes,
    draw_ground_plane
)


def main():
    board = create_charuco_board()
    detector_params = cv2.aruco.DetectorParameters()

    # Load calibration
    camera_matrix, dist_coeffs = load_calibration("./calibration/MicrosoftLifeCam_fixedFocus50_calib.npz")
    if camera_matrix is None:
        print("ERROR: No calibration file found. Run calibrate_camera.py first.")
        return
    else:
        print("Calibration loaded. Starting OpenGL renderer...")

    # Initialize Pygame + OpenGL
    width = 1280
    height = 720
    pygame.init()
    pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("AR Cube - PyOpenGL")
    
    setup_projection(camera_matrix, width, height)
    
    glEnable(GL_DEPTH_TEST)
    glClearColor(0, 0, 0, 1)

    with ExitStack() as stack:
        cap = stack.enter_context(openCapture(0, width, height))
        stack.callback(pygame.quit)

        zero_rvec = zero_tvec = None

        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    return
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, board.getDictionary(), parameters=detector_params)

            pose_found = False
            rvec = tvec = None
            charuco_corners = charuco_ids = None
                       
            if ids is not None and len(ids) > 0:
                # Interpolate ChArUco corners
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=board,
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs
                )

                if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
                    pose_res = cv2.aruco.estimatePoseCharucoBoard(
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids,
                        board=board,
                        cameraMatrix=camera_matrix,
                        distCoeffs=dist_coeffs,
                        rvec=None,
                        tvec=None
                    )
                    if zero_rvec is None and zero_tvec is None:
                        zero_rvec = rvec
                        zero_tvec = tvec
                    if isinstance(pose_res, tuple):
                        if len(pose_res) == 3:
                            ok, rvec, tvec = pose_res
                            pose_found = ok and (rvec is not None) and (tvec is not None)
                        elif len(pose_res) == 2:
                            rvec, tvec = pose_res
                            pose_found = True
                
            # Clear and render
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()

            projection_matrix = calculateGlProjection(camera_matrix, width, height, 0.1, 100.0)
            glMultTransposeMatrixf(projection_matrix)
            
            glViewport(0, 0, width, height)

            # Draw camera frame as background texture
            glDisable(GL_DEPTH_TEST)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_flip = cv2.flip(frame_rgb, 0)
            glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, frame_flip)
            glEnable(GL_DEPTH_TEST)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            if pose_found:
                R, _ = cv2.Rodrigues(rvec)
                mv = np.eye(4) 
                mv[:3, :3] = R
                mv[:3, 3] = tvec.flatten()
                z_coor = np.eye(4)
                z_coor[2,2] = -1  # Invert Z axis
                glMultTransposeMatrixf(z_coor @ mv)

            
            glEnable(GL_LINE_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

            draw_ground_plane(start=(-0.5, -0.5), end=(0.5, 0.5), segments=20)
            draw_axes(size=0.15)
            if pose_found:
                draw_cube_gl(size=0.05)
            
            pygame.display.flip()
            clock.tick(30)  # 30 FPS


if __name__ == "__main__":
    main()