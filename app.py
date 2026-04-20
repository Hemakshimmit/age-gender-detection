import os
import requests
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ===============================
# CONFIG
# ===============================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ===============================
# SAFE DOWNLOAD FUNCTION
# ===============================
def download_file(url, path):
    if not os.path.exists(path):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception:
            st.error(f"❌ Failed downloading: {path}")
            st.stop()

# ===============================
# DOWNLOAD ALL MODELS ONCE
# ===============================
MODEL_FILES = {
    "models/opencv_face_detector.pbtxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",

    "models/opencv_face_detector_uint8.pb":
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb",

    "models/age_net.caffemodel":
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",

    "models/age_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",

    "models/gender_net.caffemodel":
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel",

    "models/gender_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
}

for path, url in MODEL_FILES.items():
    download_file(url, path)

# ===============================
# VERIFY MODELS (CRITICAL FIX)
# ===============================
for file in MODEL_FILES.keys():
    if not os.path.exists(file):
        st.error(f"❌ Missing file: {file}")
        st.stop()

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Age & Gender Detection", layout="wide")

st.title("🧠 Age & Gender Detection")

st.markdown("Deep Learning · OpenCV · Streamlit")

# ===============================
# MODEL PATHS
# ===============================
FACE_PROTO   = "models/opencv_face_detector.pbtxt"
FACE_MODEL   = "models/opencv_face_detector_uint8.pb"
AGE_PROTO    = "models/age_deploy.prototxt"
AGE_MODEL    = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

GENDER_LIST = ['Male', 'Female']

# ===============================
# LOAD MODELS (SAFE)
# ===============================
@st.cache_resource
def load_models():
    try:
        face = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
        age = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        gender = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        return face, age, gender
    except Exception as e:
        st.error("❌ OpenCV model loading failed")
        st.exception(e)
        st.stop()

face_net, age_net, gender_net = load_models()

# ===============================
# FACE DETECTION
# ===============================
def detect_faces(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], False)
    face_net.setInput(blob)
    detections = face_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.7:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((x1, y1, x2, y2))
    return boxes

# ===============================
# PREDICTION
# ===============================
def predict(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                 (78.4, 87.7, 114.8), swapRB=False)

    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward()[0].argmax()]

    age_net.setInput(blob)
    age = AGE_BUCKETS[age_net.forward()[0].argmax()]

    return gender, age

# ===============================
# PROCESS IMAGE
# ===============================
def process(img):
    output = img.copy()
    boxes = detect_faces(img)

    results = []

    for (x1, y1, x2, y2) in boxes:
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        gender, age = predict(face)

        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(output, f"{gender}, {age}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        results.append((gender, age))

    return output, results

# ===============================
# UI
# ===============================
file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = np.array(Image.open(file))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    result, res = process(img)

    col1, col2 = st.columns(2)

    with col1:
        st.image(file, caption="Original")

    with col2:
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                 caption="Detected")

    st.success(f"Faces detected: {len(res)}")

    for i, r in enumerate(res):
        st.write(f"Face {i+1}: {r}")

# ===============================
# FOOTER (YOUR STYLE BACK)
# ===============================
st.markdown("---")
st.markdown("<center>Thank You !!</center>", unsafe_allow_html=True)
