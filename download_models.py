"""
download_models.py
──────────────────
Run this script once to download all required pre-trained model files
into the `models/` directory.

Usage:
    python download_models.py
"""

import os
import urllib.request

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    # Face detection (TensorFlow)
    "opencv_face_detector.pbtxt": (
        "https://raw.githubusercontent.com/spmallick/learnopencv/"
        "master/AgeGender/opencv_face_detector.pbtxt"
    ),
    "opencv_face_detector_uint8.pb": (
        "https://github.com/spmallick/learnopencv/raw/"
        "master/AgeGender/opencv_face_detector_uint8.pb"
    ),
    # Age estimation (Caffe)
    "age_deploy.prototxt": (
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/"
        "master/age_gender_models/age_deploy.prototxt"
    ),
    "age_net.caffemodel": (
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/"
        "master/models/age_net.caffemodel"
    ),
    # Gender classification (Caffe)
    "gender_deploy.prototxt": (
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/"
        "master/age_gender_models/gender_deploy.prototxt"
    ),
    "gender_net.caffemodel": (
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/"
        "master/models/gender_net.caffemodel"
    ),
}


def download(filename, url):
    dest = os.path.join(MODEL_DIR, filename)
    if os.path.exists(dest):
        print(f"  ✓ Already exists: {filename}")
        return
    print(f"  ↓ Downloading {filename} …", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"done ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"FAILED → {e}")
        print(f"    Manual URL: {url}")


if __name__ == "__main__":
    print("=" * 55)
    print(" Downloading Age & Gender Detection Model Files")
    print("=" * 55)
    for fname, url in FILES.items():
        download(fname, url)
    print("\nAll done! Run the app with:  streamlit run app.py")
