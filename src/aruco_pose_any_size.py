import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)

MARKER_SIZE = 0.05  # meters (change to real size)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

focal = w
camera_matrix = np.array([
    [focal, 0, w / 2],
    [0, focal, h / 2],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros((5, 1))

half = MARKER_SIZE / 2
object_points = np.array([
    [-half,  half, 0],
    [ half,  half, 0],
    [ half, -half, 0],
    [-half, -half, 0]
], dtype=np.float32)

ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11
}

detectors = {}
for name, dict_id in ARUCO_DICTS.items():
    d = aruco.getPredefinedDictionary(dict_id)
    p = aruco.DetectorParameters()
    detectors[name] = aruco.ArucoDetector(d, p)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for name, detector in detectors.items():
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)

            for i in range(len(ids)):
                img_pts = corners[i].reshape(4, 2)

                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    img_pts,
                    camera_matrix,
                    dist_coeffs
                )

                if success:
                    cv2.drawFrameAxes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        MARKER_SIZE / 2
                    )

            cv2.putText(
                frame,
                f"Pose: {name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            break

    cv2.imshow("ArUco Pose Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

