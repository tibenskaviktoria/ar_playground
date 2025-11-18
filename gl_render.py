# Create a program that uses the view from the camera and draws a 3D cube on a detected ArUco marker.
from contextlib import ExitStack
import cv2
import numpy as np
import pygame
import pyrr
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from cv_utils import (
    load_calibration,
    create_charuco_board,
    openCapture
)


def transform_from_rvec_tvec(rvec, tvec):
    """Convert OpenCV rvec/tvec to OpenGL 4x4 model-view matrix."""
    R, _ = cv2.Rodrigues(rvec)
    # Build 4x4 matrix: [R | t]
    #                   [0 | 1]
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = R
    matrix[:3, 3] = tvec.flatten()
    return matrix.T.flatten()  # OpenGL uses column-major (transpose for row-major input)


def draw_cube_gl(size=0.02):
    """Draw a wireframe cube using OpenGL with bottom-left corner at origin."""
    # Shift the cube so its bottom-left corner (was at [-size,-size,0]) moves to (0,0,0)
    glPushMatrix()
    glTranslatef(size + 0.03, size + 0.03, 0.0)
    
    glColor3f(1, 1, 1)  # White
    glLineWidth(3.0)

    glBegin(GL_LINES)
    
    vertices = [
        [-size, -size, 0],
        [size, -size, 0],
        [size, size, 0],
        [-size, size, 0],
        [-size, -size, -size * 2],
        [size, -size, -size * 2],
        [size, size, -size * 2],
        [-size, size, -size * 2]
    ]
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    for start, end in edges:
        glVertex3fv(vertices[start])
        glVertex3fv(vertices[end])
    
    glEnd()
    
    # Draw vertices as points
    glPointSize(10.0)
    glBegin(GL_POINTS)
    glColor3f(1, 0, 0)  # Red
    for v in vertices:
        glVertex3fv(v)
    glEnd()

    glLineWidth(1.0)
    glPopMatrix()


def setup_projection(camera_matrix, width, height, near=0.01, far=100):
    """Set up OpenGL projection matrix from camera intrinsics."""
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    
    # OpenGL frustum: (left, right, bottom, top, near, far)
    glFrustum(
        -cx / fx * near,
        (width - cx) / fx * near,
        (height - cy) / fy * near,
        -cy / fy * near,
        near,
        far
    )
    glMatrixMode(GL_MODELVIEW)


def calculateGlProjection(camera_matrix, width, height, z_near, z_far):
    # from http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    cx = camera_matrix[0,2]
    cy = camera_matrix[1,2]
    m = np.zeros((4,4))
    m[0,0] = fx 
    m[1,1] = fy
    m[0,2] = cx
    m[1,2] = cy
    m[2,2] = -(z_far+z_near)
    m[3,2] = 1
    m[2,3] = z_far*z_near
    m[:,2] *= -1.0
    ortho = pyrr.matrix44.create_orthogonal_projection(0, width, height, 0, z_near, z_far)
    return ortho.T @ m


def draw_ground_plane(start, end, segments):
    glColor3f(0.5, 0.5, 0.5)
    glLineWidth(0.2)
    glBegin(GL_LINES)
    x_s = start[0]
    x_e = end[0]
    for y in np.linspace(start=start[1], stop=end[1], num=segments):
        glVertex3f(x_s, y, 0.0)  
        glVertex3f(x_e, y, 0.0)
    y_s = start[1]
    y_e = end[1]        
    for x in np.linspace(start=start[0], stop=end[0], num=segments):
        glVertex3f(x, y_s, 0.0)  
        glVertex3f(x, y_e, 0.0)
    glEnd()
    glLineWidth(1.0)


def draw_axes(size):
    glPushMatrix()
    glLineWidth(3.0) # set desired line width (in pixels)

    # X: Red
    glColor3f(1, 0, 0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)  
    glVertex3f(size, 0.0, 0.0)
    glEnd()

    # Y: Green
    glColor3f(0, 1, 0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0) 
    glVertex3f(0.0, size, 0.0)
    glEnd()

    # Z: Blue
    glColor3f(0, 0, 1)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, size)
    glEnd()
    glPopMatrix()

    glLineWidth(1.0) # reset to default line width


def main():
    board = create_charuco_board()
    detector_params = cv2.aruco.DetectorParameters()

    # Load calibration
    camera_matrix, dist_coeffs = load_calibration("MicrosoftLifeCam_fixedFocus50_calib.npz")
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