
import os
import urllib.request

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def download_file(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)

# Face Detection
download_file(
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",
    "models/opencv_face_detector.pbtxt"
)

download_file(
    "https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb",
    "models/opencv_face_detector_uint8.pb"
)

# Age Model
download_file(
    "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
    "models/age_net.caffemodel"
)

download_file(
    "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_deploy.prototxt",
    "models/age_deploy.prototxt"
)

# Gender Model
download_file(
    "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel",
    "models/gender_net.caffemodel"
)

download_file(
    "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_deploy.prototxt",
    "models/gender_deploy.prototxt"
)
    
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import os

# ─────────────────────────────────────────────────
#  Page configuration (must be first Streamlit call)
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Age & Gender Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────
#  Custom CSS – clean, modern UI
# ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

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
#  Model paths  (place model files inside /models/)
# ─────────────────────────────────────────────────
MODEL_DIR = "models"

FACE_PROTO   = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL   = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO    = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL    = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")

# ─────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────
AGE_BUCKETS    = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                  '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST    = ['Male', 'Female']
MODEL_MEAN_VAL = (78.4263377603, 87.7689143744, 114.895847746)
CONF_THRESHOLD = 0.7   # face-detection confidence threshold


# ─────────────────────────────────────────────────
#  Load models (cached so they load only once)
# ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load face, age, and gender DNN models into memory."""
    missing = []
    for path in [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL,
                 GENDER_PROTO, GENDER_MODEL]:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        return None, None, None, missing

    face_net   = cv2.dnn.readNet(FACE_MODEL,   FACE_PROTO)
    age_net    = cv2.dnn.readNet(AGE_MODEL,    AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return face_net, age_net, gender_net, []


# ─────────────────────────────────────────────────
#  Core detection functions
# ─────────────────────────────────────────────────
def detect_faces(net, frame, conf_threshold=CONF_THRESHOLD):
    """
    Run face detection on a frame.
    Returns list of bounding boxes as (x1, y1, x2, y2).
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 [104, 117, 123], swapRB=False)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold:
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))
            boxes.append((x1, y1, x2, y2))
    return boxes


def predict_age(net, face_img):
    """Predict age bucket for a cropped face image."""
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 MODEL_MEAN_VAL, swapRB=False)
    net.setInput(blob)
    preds = net.forward()
    return AGE_BUCKETS[preds[0].argmax()]


def predict_gender(net, face_img):
    """Predict gender for a cropped face image."""
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                 MODEL_MEAN_VAL, swapRB=False)
    net.setInput(blob)
    preds = net.forward()
    return GENDER_LIST[preds[0].argmax()]


def process_frame(frame, face_net, age_net, gender_net, padding=20):
    """
    Run full pipeline on a single BGR frame.
    Returns annotated frame + list of result dicts.
    """
    results = []
    output  = frame.copy()
    h, w    = frame.shape[:2]
    boxes   = detect_faces(face_net, frame)

    for (x1, y1, x2, y2) in boxes:
        # Pad the face crop slightly for better accuracy
        x1p = max(0, x1 - padding)
        y1p = max(0, y1 - padding)
        x2p = min(w, x2 + padding)
        y2p = min(h, y2 + padding)
        face_crop = frame[y1p:y2p, x1p:x2p]

        if face_crop.size == 0:
            continue

        gender = predict_gender(gender_net, face_crop)
        age    = predict_age(age_net, face_crop)

        label = f"{gender}, Age: {age}"
        results.append({"gender": gender, "age": age,
                         "box": (x1, y1, x2, y2)})

        # Draw bounding box
        color = (147, 112, 219) if gender == "Male" else (255, 105, 180)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(output, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return output, results


# ─────────────────────────────────────────────────
#  Helper – check model availability with nice UI
# ─────────────────────────────────────────────────
def show_model_missing_error(missing_files):
    st.error("Model files not found! Please download and place them in the `models/` folder.")
    st.markdown("**Missing files:**")
    for f in missing_files:
        st.code(f)
    st.markdown("""
### Download Model Files

Run these commands in your terminal:

```bash
mkdir models && cd models

# Face Detection
wget https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt
wget https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb

# Age Estimation
wget https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel
wget https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_gender_models/age_deploy.prototxt

# Gender Classification
wget https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel
wget https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_gender_models/gender_deploy.prototxt
```
""")


# ─────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.markdown("### ⚙️ Settings")
    conf_thresh = st.slider("Face Detection Confidence", 0.3, 0.95, 0.7, 0.05)
    padding     = st.slider("Face Crop Padding (px)", 0, 50, 20, 5)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
- **Developers**: Hemakshi Ingale
                  Prabhanjan Ingle  
                  Divya Dosi
- **Special Thanks**: Prof. Yamini Warke  
- **MMIT College, Pune**  
""")
    st.markdown("---")
    st.caption("Deep Learning Mini Project · 2025-26")


# ─────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────
st.markdown('<div class="main-title">Age & Gender Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Deep Learning · OpenCV DNN · Caffe Models · Streamlit</div>',
            unsafe_allow_html=True)

# Load models
with st.spinner("Loading deep learning models…"):
    face_net, age_net, gender_net, missing = load_models()

if missing:
    show_model_missing_error(missing)
    st.stop()

st.success("Ready to scan")

# ─────────────────────────────────────────────────
#  Mode selector
# ─────────────────────────────────────────────────
st.markdown("---")
mode = st.radio("**Select Input Mode**", ["Upload Image", "Webcam (Real-Time)"],
                horizontal=True)
st.markdown("---")


# ══════════════════════════════════════════════════
#  MODE 1 – Image Upload
# ══════════════════════════════════════════════════
if mode == "Upload Image":
    st.markdown("### Upload an Image")
    uploaded = st.file_uploader("Choose a JPG / PNG / JPEG image",
                                type=["jpg", "jpeg", "png"])

    if uploaded:
        # Decode uploaded file → numpy BGR frame
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            # Display original in RGB
            st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                     use_column_width=True)

        with col2:
            st.markdown("**Detected Faces with Labels**")
            with st.spinner("Running detection…"):
                out_frame, results = process_frame(
                    frame_bgr, face_net, age_net, gender_net, padding=padding)

            st.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB),
                     use_column_width=True)

        # Results summary
        st.markdown("---")
        st.markdown(f"###  Detection Results — {len(results)} face(s) found")
        if results:
            cols = st.columns(min(len(results), 4))
            for idx, res in enumerate(results):
                with cols[idx % 4]:
                    icon = "👨" if res["gender"] == "Male" else "👩"
                    st.markdown(f"""
<div class="result-card">
  <b>Face #{idx + 1}</b><br>
  {icon} <b>Gender:</b> {res['gender']}<br>
   <b>Age:</b> {res['age']}
</div>""", unsafe_allow_html=True)
        else:
            st.warning("No faces detected. Try lowering the confidence threshold in the sidebar.")
    else:
        st.markdown('<div class="info-box">Upload an image above to get started.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════
#  MODE 2 – Webcam Real-Time
# ══════════════════════════════════════════════════
else:
    st.markdown("###  Real-Time Webcam Detection")
    st.markdown('<div class="info-box">Click <b>Start Webcam</b> to begin. '
                'Press <b>Stop</b> to end the session.</div>',
                unsafe_allow_html=True)

    run       = st.checkbox("Start Webcam")
    frame_ph  = st.empty()   # placeholder for video frames
    result_ph = st.empty()   # placeholder for live results

    if run:
        cap = cv2.VideoCapture(0)   # 0 = default webcam

        if not cap.isOpened():
            st.error("Could not access webcam. "
                     "Make sure your browser has camera permission and "
                     "no other app is using it.")
        else:
            st.info(" Webcam active — uncheck the box above to stop.")
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to grab frame.")
                    break

                frame = cv2.flip(frame, 1)   # mirror effect
                out_frame, results = process_frame(
                    frame, face_net, age_net, gender_net, padding=padding)

                # Show annotated frame
                frame_ph.image(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB),
                                channels="RGB", use_column_width=True)

                # Show live text results
                if results:
                    lines = " | ".join(
                        [f"Face {i+1}: {r['gender']}, {r['age']}"
                         for i, r in enumerate(results)]
                    )
                    result_ph.success(f" {lines}")
                else:
                    result_ph.info("No faces detected in this frame…")

                # Re-read checkbox state to allow stopping
                run = st.session_state.get("▶ Start Webcam", True)

            cap.release()
    else:
        st.info("Check the box above to start the webcam feed.")


# ─────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Thank You!!</small></center>",
    unsafe_allow_html=True,
)
