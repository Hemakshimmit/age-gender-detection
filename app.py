import os
import requests
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────
# MODEL DOWNLOAD (SAFE FOR STREAMLIT CLOUD)
# ─────────────────────────────────────────────────
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url, path):
    if not os.path.exists(path):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
        except Exception as e:
            st.error(f"Model download failed: {path}")
            st.stop()

# Download once only
if not os.path.exists("models/opencv_face_detector_uint8.pb"):

    st.info("🔄 Downloading AI models for first run... please wait")

    download_file(
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",
        "models/opencv_face_detector.pbtxt"
    )

    download_file(
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb",
        "models/opencv_face_detector_uint8.pb"
    )

    download_file(
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
        "models/age_net.caffemodel"
    )

    download_file(
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_deploy.prototxt",
        "models/age_deploy.prototxt"
    )

    download_file(
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel",
        "models/gender_net.caffemodel"
    )

    download_file(
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_deploy.prototxt",
        "models/gender_deploy.prototxt"
    )

# ─────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Age & Gender Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# YOUR ORIGINAL UI STYLE (RESTORED)
# ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .info-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.2rem;
        border-left: 4px solid #667eea;
    }

    .result-card {
        background: #fff;
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #764ba2;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────
st.markdown('<div class="main-title">Age & Gender Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning · OpenCV DNN · Caffe Models · Streamlit</div>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────────
FACE_PROTO   = "models/opencv_face_detector.pbtxt"
FACE_MODEL   = "models/opencv_face_detector_uint8.pb"
AGE_PROTO    = "models/age_deploy.prototxt"
AGE_MODEL    = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']

GENDER_LIST = ['Male', 'Female']

# ─────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────
@st.cache_resource
def load_models():
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return face_net, age_net, gender_net

face_net, age_net, gender_net = load_models()

# ─────────────────────────────────────────────────
# DETECTION FUNCTIONS
# ─────────────────────────────────────────────────
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


def predict(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                 (78.4, 87.7, 114.8), swapRB=False)

    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward()[0].argmax()]

    age_net.setInput(blob)
    age = AGE_BUCKETS[age_net.forward()[0].argmax()]

    return gender, age


def process_image(img):
    output = img.copy()
    boxes = detect_faces(img)
    results = []

    for (x1, y1, x2, y2) in boxes:
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        gender, age = predict(face)
        label = f"{gender}, {age}"

        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(output, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        results.append(label)

    return output, results

# ─────────────────────────────────────────────────
# SIDEBAR (RESTORED ABOUT SECTION)
# ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.markdown("### ⚙️ Settings")

    st.markdown("---")

    st.markdown("### 👨‍💻 About")
    st.markdown("""
- Developers:  
  Hemakshi Ingale  
  Prabhanjan Ingle  
  Divya Dosi  

- MMIT College, Pune  
- Guide: Prof. Yamini Warke  
""")

    st.markdown("---")
    st.caption("Deep Learning Mini Project · 2025-26")

# ─────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────
mode = st.radio("Select Input Mode", ["Upload Image"])

if mode == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        img = np.array(Image.open(file))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        result_img, results = process_image(img)

        col1, col2 = st.columns(2)

        with col1:
            st.image(file, caption="Original")

        with col2:
            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                     caption="Detected")

        st.success(f"Faces detected: {len(results)}")

        for i, r in enumerate(results):
            st.write(f"Face {i+1}: {r}")

# ─────────────────────────────────────────────────
# FOOTER (RESTORED)
# ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("<center><small>Thank You!!</small></center>", unsafe_allow_html=True)
