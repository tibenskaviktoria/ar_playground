from pathlib import Path
import cv2
from contextlib import ExitStack
from cv_utils import (
    openCapture, 
    load_calibration, 
    save_calibration, 
    create_charuco_board,
    createWindow
)
from utils import timestamp


def main():
    board = create_charuco_board()
    detector_params = cv2.aruco.DetectorParameters()

    # Storage for calibration captures
    all_corners = []   # list of charuco corners (per image)
    all_ids = []       # list of charuco ids (per image)
    img_size = None

    # Try to load existing calibration
    camera_matrix, dist_coeffs = load_calibration("MicrosoftLifeCam_fixedFocus50_calib.npz")
    if camera_matrix is None:
        print("No calib.npz found — running in uncalibrated mode. Press 'c' to capture charuco frames, 'k' to calibrate.")
    else:
        print("Loaded calib.npz — pose estimation enabled.")

    dir = Path(f"captures{timestamp()}")
    dir.mkdir(exist_ok = False)


    # Enter window + capture contexts with a single ExitStack (avoids deep nesting)
    with ExitStack() as stack:
        stack.enter_context(createWindow("Camera (ChArUco)"))
        cap = stack.enter_context(openCapture(0))
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            if img_size is None:
                h, w = frame.shape[:2]
                img_size = (w, h)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, board.getDictionary(), parameters=detector_params)

            charuco_corners = None
            charuco_ids = None
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

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
                    # print(f"Detected {len(charuco_ids)} Charuco corners")
                    cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 0))

                    # If calibrated, estimate pose
                    if camera_matrix is not None and dist_coeffs is not None:
                        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            charucoCorners=charuco_corners,
                            charucoIds=charuco_ids,
                            board=board,
                            cameraMatrix=camera_matrix,
                            distCoeffs=dist_coeffs,
                            rvec =None,
                            tvec =None
                        )
                        if ok:
                            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, board.getSquareLength() * 3)

            # UI text
            mode = "CALIBRATED" if camera_matrix is not None else "UNCALIBRATED"
            cv2.putText(frame, f"Mode: {mode}  Captures: {len(all_corners)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Camera (ChArUco)", frame)

            key = cv2.waitKey(1) & 0xFF
            # Quit
            if key == ord('q'):
                break

            # Capture current charuco corners (only if we detected charuco corners)
            if key == ord('c'):
                if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 3:
                    # store as required by calibrateCameraCharuco: list of arrays, and list of ids
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    cv2.imwrite(dir.joinpath(f"image{len(all_corners)}.png"), frame)
                    print(f"Captured frame #{len(all_corners)} (charuco corners: {len(charuco_ids)})")
                else:
                    print("No valid Charuco corners detected to capture (need >3). Move board and try again.")

            # Run calibration from captured frames
            if key == ord('k'):
                if len(all_corners) < 3:
                    print("Need at least 3 captures to run calibration. Captured:", len(all_corners))
                else:
                    print("Running calibration with", len(all_corners), "frames...")
                    try:
                        ret, cam_mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                            charucoCorners=all_corners,
                            charucoIds=all_ids,
                            board=board,
                            imageSize=img_size,
                            cameraMatrix=None,
                            distCoeffs=None
                        )
                    except Exception as e:
                        print("Calibration failed:", e)
                        continue

                    if ret:
                        camera_matrix, dist_coeffs = cam_mtx, dist
                        save_calibration(camera_matrix, dist_coeffs, "calib.npz")
                        print("Calibration RMS error:", ret)
                    else:
                        print("Calibration did not return a valid result.")

            # Reset captured frames
            if key == ord('r'):
                all_corners = []
                all_ids = []
                print("Cleared captured frames.")

if __name__ == "__main__":
    main()