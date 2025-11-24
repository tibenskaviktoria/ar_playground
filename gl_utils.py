import numpy as np
import cv2
import pyrr
from OpenGL.GL import *


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
