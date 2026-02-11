import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0) # for taking the live video from the camera

ARUCO_DICTS = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
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

            cv2.putText(
                frame,
                f"Detected: {name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
            break

    cv2.imshow("ArUco Detection (Any Size)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

