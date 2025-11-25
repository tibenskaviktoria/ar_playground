import os
import numpy as np
import cv2
from contextlib import contextmanager

@contextmanager
def createWindow(title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    try:
        yield
    finally:
        cv2.destroyWindow(title)


@contextmanager
def openCapture(camera_index=1, image_width=1280, image_height=720):
    # Open default camera (0). On Windows, CAP_DSHOW often reduces latency when opening.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    # Optional: request a resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    # Disable auto-focus for stable calibration
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 50)  # set focus (adjust as needed)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=auto exposure

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    try:
        yield cap
    finally:
        cap.release()


def load_calibration(path="calib.npz"):
    if not os.path.exists(path):
        return None, None
    d = np.load(path)
    return d["camera_matrix"], d["dist_coeffs"]

#FIXME: save image-size along the calibration parameters!
def save_calibration(camera_matrix, dist_coeffs, path="calib.npz"):
    np.savez(path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Saved calibration to {path}")


def create_charuco_board(scale = 1):
    # --- ChArUco board parameters (adjust to your printed board) ---
    squares_x = 7   # number of chessboard squares in X direction
    squares_y = 5   # number of chessboard squares in Y direction
    square_length = 0.03 * scale  # meters (or any other unit) - used for pose/scale
    marker_length = 0.022 * scale # must be < square_length

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    return board


def transfrom_rvec_tvec_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = tvec.flatten()
    return T


def estimate_pose_charuco(gray_image, board, camera_matrix, dist_coeffs):
    MIN_CORNERS_FOR_POSE = 4

    corners, ids, _ = cv2.aruco.detectMarkers(gray_image, board.getDictionary())
    if ids is None or len(ids) < MIN_CORNERS_FOR_POSE:
        return None, None
    
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray_image,
        board=board
    )

    if charuco_ids is None or len(charuco_ids) < MIN_CORNERS_FOR_POSE:
        return None, None
    
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids,
        board=board,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        rvec=None,
        tvec=None
    )

    if not retval:
        return None, None
    
    return transfrom_rvec_tvec_to_matrix(rvec, tvec), rvec, tvec