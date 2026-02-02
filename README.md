# OpenCV ArUco Pose Estimation (Modern API)

A clean and up-to-date OpenCV project for detecting **any ArUco marker size**
(4×4, 5×5, 6×6, 7×7, ArUco Original, AprilTags) and estimating their **3D pose**
using a live camera feed.

This repository uses the **latest OpenCV ArUco API (OpenCV ≥ 4.7)** and avoids
deprecated functions.

---

## ✨ Features
- Detect **any ArUco marker size**
- Supports **AprilTags**
- Live camera detection
- 3D pose estimation (rotation & translation)
- Uses modern `ArucoDetector` + `solvePnP`
- Beginner-friendly & well-structured
- No deprecated OpenCV functions

---

## 📦 Requirements
- Python 3.8+
- OpenCV ≥ 4.7
- NumPy

Install dependencies:
```bash
pip install -r requirements.txt
