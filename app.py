import os
import requests
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# =========================
# MODEL SETUP
# =========================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url, path):
    if not os.path.exists(path):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)

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

for p, u in MODEL_FILES.items():
    download_file(u, p)

# =========================
# CHECK FILES
# =========================
for f in MODEL_FILES.keys():
    if not os.path.exists(f):
        st.error(f"Missing: {f}")
        st.stop()

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Age & Gender Detection",
    page_icon="🧠",
    layout="wide"
)

# =========================
# UI DESIGN (YOUR ORIGINAL STYLE)
# =========================
st.markdown("""
<style>
.main-title {
    font-size: 2.6rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg,#667eea,#764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-title {
    text-align: center;
    color: gray;
    margin-bottom: 20px;
}
.result-card {
    padding: 10px;
    background: #fff;
    border-left: 4px solid #764ba2;
    margin: 5px 0;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Age & Gender Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning · OpenCV DNN · Streamlit</div>', unsafe_allow_html=True)

# =========================
# MODEL PATHS
# =========================
FACE_PROTO = "models/opencv_face_detector.pbtxt"
FACE_MODEL = "models/opencv_face_detector_uint8.pb"
AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"
GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

AGE_BUCKETS = ['(0-2)','(4-6)','(8-12)','(15-20)',
               '(25-32)','(38-43)','(48-53)','(60-100)']

GENDER = ["Male", "Female"]

# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    face = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return face, age, gender

face_net, age_net, gender_net = load_models()

# =========================
# FUNCTIONS
# =========================
def detect_faces(img):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), [104,117,123], False)
    face_net.setInput(blob)
    det = face_net.forward()

    boxes = []
    for i in range(det.shape[2]):
        if det[0,0,i,2] > 0.7:
            x1 = int(det[0,0,i,3]*w)
            y1 = int(det[0,0,i,4]*h)
            x2 = int(det[0,0,i,5]*w)
            y2 = int(det[0,0,i,6]*h)
            boxes.append((x1,y1,x2,y2))
    return boxes


def predict(face):
    blob = cv2.dnn.blobFromImage(face,1,(227,227),(78.4,87.7,114.8),False)

    gender_net.setInput(blob)
    gender = GENDER[gender_net.forward()[0].argmax()]

    age_net.setInput(blob)
    age = AGE_BUCKETS[age_net.forward()[0].argmax()]

    return gender, age


def process(img):
    out = img.copy()
    boxes = detect_faces(img)
    results = []

    for (x1,y1,x2,y2) in boxes:
        face = img[y1:y2,x1:x2]
        if face.size == 0:
            continue

        g,a = predict(face)

        cv2.rectangle(out,(x1,y1),(x2,y2),(255,0,255),2)
        cv2.putText(out,f"{g},{a}",(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        results.append((g,a))

    return out, results

# =========================
# SIDEBAR (ABOUT RESTORED)
# =========================
with st.sidebar:
    st.markdown("### About Developers")

    st.markdown("""
**Developers:**  
- Hemakshi Ingale  
- Prabhanjan Ingle  
- Divya Dosi  

**Special Thanks:**  
- Prof. Yamini Warke  

MMIT College, Pune  
""")

# =========================
# MODE SELECT
# =========================
mode = st.radio("Select Mode", ["Upload Image", "Webcam"])

# =========================
# IMAGE MODE
# =========================
if mode == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = np.array(Image.open(file))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        out,res = process(img)

        c1,c2 = st.columns(2)

        with c1:
            st.image(file)

        with c2:
            st.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB))

        st.success(f"Faces: {len(res)}")

        for i,r in enumerate(res):
            st.markdown(f"""
            <div class="result-card">
            Face {i+1}: {r}
            </div>
            """, unsafe_allow_html=True)

# =========================
# WEBCAM MODE (FIXED)
# =========================
else:
    st.markdown("### 🎥 Webcam Detection")

    picture = st.camera_input("Take a picture")

    if picture:
        img = np.array(Image.open(picture))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        out, res = process(img)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Captured Image")

        with col2:
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB),
                     caption="Detected Output")

        st.success(f"Faces detected: {len(res)}")

        for i, r in enumerate(res):
            st.write(f"Face {i+1}: {r}")

# =========================
# FOOTER (RESTORED)
# =========================
st.markdown("---")
st.markdown("<center>Thank You !!</center>", unsafe_allow_html=True)
