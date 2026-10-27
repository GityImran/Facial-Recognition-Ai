"""
Minimalist Face Emotion Recognition System
Supports Live Stream (WebRTC), Browser Snapshot, Host Webcam, and Image Upload.
"""
import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import timm
import os
import time
import textwrap

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

RTC_CONFIGURATION = (
    RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
    )
    if WEBRTC_AVAILABLE
    else None
)

def clean_html(s: str) -> str:
    """Safely strips leading indentation to prevent markdown <pre> formatting."""
    return textwrap.dedent(s).strip()

# ─── Page Config ───
st.set_page_config(
    page_title="Emotion AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Constants ───
CLASSES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
EMOJI_MAP = {
    "Surprise": "😲",
    "Fear": "😨",
    "Disgust": "🤢",
    "Happiness": "😊",
    "Sadness": "😢",
    "Anger": "😠",
    "Neutral": "😐"
}
# Contemporary muted palette: Amber, Pink, Violet, Emerald, Blue, Rose, Slate
COLOR_MAP = {
    "Surprise": (245, 158, 11),   # #f59e0b
    "Fear": (236, 72, 153),       # #ec4899
    "Disgust": (168, 85, 247),    # #a855f7
    "Happiness": (16, 185, 129),  # #10b981
    "Sadness": (59, 130, 246),    # #3b82f6
    "Anger": (239, 68, 68),       # #ef4444
    "Neutral": (148, 163, 184)    # #94a3b8
}

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Definitions ───
def build_resnet50(num_classes=7):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

def build_efficientnet(num_classes=7):
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
    return nn.Sequential(
        model,
        nn.Dropout(0.4),
        nn.Linear(model.num_features, 512),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

def build_vit(num_classes=7):
    return timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)

MODEL_FILES = {
    "ResNet50": "rafdb_model_resnet50.pth",
    "EfficientNetB3": "rafdb_model_efficientnetb3.pth",
    "ViT-Small": "rafdb_model_vit-small.pth"
}

MODEL_BUILDERS = {
    "ResNet50": lambda: build_resnet50(7),
    "EfficientNetB3": lambda: build_efficientnet(7),
    "ViT-Small": lambda: build_vit(7)
}

@st.cache_resource
def load_model(name):
    path = os.path.join(MODEL_DIR, MODEL_FILES[name])
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = MODEL_BUILDERS[name]()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(DEVICE).eval()

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_face(face_img, model_name=None):
    pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return img_transform(pil).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict(model, tensor):
    probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs[idx], probs

# ─── Minimalist CSS ───
st.markdown(clean_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #cbd5e1;
}

.stApp {
    background-color: #090b10;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1140px !important;
}

header[data-testid="stHeader"] {
    background-color: transparent !important;
}

[data-testid="stSidebar"] {
    background-color: #0e1118 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

[data-testid="stSidebar"] h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.01em;
    margin-bottom: 0.8rem;
}

/* Header Container */
.brand-container {
    padding: 0 0 1.25rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #818cf8;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    padding: 3px 8px;
    border-radius: 4px;
    margin-bottom: 8px;
}

.brand-title {
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #f8fafc;
    margin: 0;
}

.brand-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 4px;
    margin-bottom: 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #11141d;
    padding: 4px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    margin-bottom: 1.25rem;
}

.stTabs [data-baseweb="tab"] {
    height: 36px;
    border-radius: 6px;
    padding: 0 16px;
    color: #94a3b8;
    font-weight: 500;
    font-size: 13.5px;
    border: none !important;
    background-color: transparent;
}

.stTabs [aria-selected="true"] {
    background-color: #1e2433 !important;
    color: #f8fafc !important;
}

.stTabs [data-baseweb="tab-border"],
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

/* Prediction Result Card */
.result-card {
    background: #11141d;
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 1.25rem;
}

.dominant-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.85rem;
    margin-bottom: 0.85rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.dominant-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.dominant-emoji {
    font-size: 2rem;
    line-height: 1;
}

.dominant-name {
    font-size: 1.25rem;
    font-weight: 600;
    color: #f8fafc;
    letter-spacing: -0.02em;
}

.dominant-pill {
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.06);
    color: #e2e8f0;
    font-variant-numeric: tabular-nums;
}

/* Probabilities */
.prob-list {
    display: flex;
    flex-direction: column;
    gap: 9px;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.prob-name {
    width: 80px;
    font-size: 12.5px;
    font-weight: 500;
    color: #94a3b8;
    white-space: nowrap;
}

.prob-track {
    flex: 1;
    height: 5px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 999px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.3s ease-out;
}

.prob-pct {
    width: 44px;
    text-align: right;
    font-size: 12px;
    color: #64748b;
    font-variant-numeric: tabular-nums;
}

.prob-row.active .prob-name,
.prob-row.active .prob-pct {
    color: #f1f5f9;
    font-weight: 600;
}

/* Empty State Card */
.empty-card {
    background: #11141d;
    border: 1px dashed rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 2.25rem 1.5rem;
    text-align: center;
}

.empty-title {
    font-size: 0.95rem;
    font-weight: 500;
    color: #94a3b8;
    margin-bottom: 4px;
}

.empty-caption {
    font-size: 0.8rem;
    color: #64748b;
    margin: 0;
}

/* Reference Category Pills */
.category-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.category-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 12px;
    background: #11141d;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    font-size: 12px;
    color: #94a3b8;
}

.category-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* Info card */
.panel-card {
    background: #11141d;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

.panel-row {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    padding: 4px 0;
    color: #94a3b8;
}

.panel-row span:last-child {
    color: #e2e8f0;
    font-weight: 500;
}
</style>
"""), unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.markdown("### Settings")
    model_name = st.selectbox(
        "Model",
        list(MODEL_FILES.keys()),
        index=0,  # Default to ResNet50
        help="Select the deep learning model architecture used for inference."
    )
    st.caption("ResNet50 delivers highest classification precision.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    st.markdown("##### Detector Parameters")
    face_scale = st.slider("Scale Factor", 1.05, 1.5, 1.3, step=0.05, help="Image pyramid scale for face bounding box detection.")
    min_neighbors = st.slider("Min Neighbors", 1, 10, 5, step=1, help="Higher values reduce false positive detections.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.expander("System & Help", expanded=False):
        st.markdown(f"**Hardware**: `{DEVICE.type.upper()}`")
        st.markdown(f"**Model Weights**: `{MODEL_FILES[model_name]}`")
        st.markdown("---")
        st.markdown("""
        - **Lighting**: Maintain even front-facing lighting.
        - **Alignment**: Face directly towards the camera lens.
        - **Sensitivity**: If no face is detected, reduce *Scale Factor* or *Min Neighbors*.
        """)

def render_emotion_stats(emotion, conf, probs, container=None):
    dominant_color = COLOR_MAP.get(emotion, (148, 163, 184))
    dom_hex = f"#{dominant_color[0]:02x}{dominant_color[1]:02x}{dominant_color[2]:02x}"

    rows_html = []
    for i, cls in enumerate(CLASSES):
        p = probs[i]
        c = COLOR_MAP.get(cls, (148, 163, 184))
        hex_c = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
        is_active = (cls == emotion)
        active_class = "active" if is_active else ""
        bar_opacity = "1" if is_active else "0.45"

        rows_html.append(
            f'<div class="prob-row {active_class}">'
            f'<span class="prob-name">{cls}</span>'
            f'<div class="prob-track"><div class="prob-fill" style="width: {p*100:.1f}%; background-color: {hex_c}; opacity: {bar_opacity};"></div></div>'
            f'<span class="prob-pct">{p:.1%}</span>'
            f'</div>'
        )

    rows_joined = "".join(rows_html)

    html = (
        f'<div class="result-card">'
        f'<div class="dominant-hero">'
        f'<div class="dominant-left">'
        f'<span class="dominant-emoji">{EMOJI_MAP.get(emotion, "😐")}</span>'
        f'<span class="dominant-name">{emotion}</span>'
        f'</div>'
        f'<span class="dominant-pill" style="border-left: 3px solid {dom_hex};">{conf:.1%} Match</span>'
        f'</div>'
        f'<div class="prob-list">{rows_joined}</div>'
        f'</div>'
    )
    if container:
        container.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)

face_detector = load_face_detector()
emotion_model = load_model(model_name)

# ─── Header ───
st.markdown(clean_html("""
<div class="brand-container">
    <div class="brand-badge">✦ Emotion AI</div>
    <h1 class="brand-title">Facial Emotion Recognition</h1>
    <p class="brand-subtitle">Real-time facial expression classification powered by PyTorch</p>
</div>
"""), unsafe_allow_html=True)

tab_live, tab_upload = st.tabs(["Camera Feed", "Image Upload"])

with tab_live:
    cam_mode = st.radio(
        "Camera Mode",
        ["Live Stream (WebRTC)", "Browser Snapshot", "Host Webcam"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if cam_mode == "Live Stream (WebRTC)":
        if not WEBRTC_AVAILABLE:
            st.markdown(clean_html("""
            <div class="empty-card" style="text-align: left; padding: 1.5rem;">
                <div class="empty-title" style="color: #f1f5f9;">Live Stream requires streamlit-webrtc</div>
                <p class="empty-caption" style="margin-top: 6px;">
                    Select <strong>Browser Snapshot</strong> above to capture frames using your camera immediately without additional libraries,
                    or run <code>pip install streamlit-webrtc av</code> to enable real-time continuous WebRTC streaming.
                </p>
            </div>
            """), unsafe_allow_html=True)
        else:
            col_cam, col_info = st.columns([1.6, 1])
            with col_cam:
                st.caption("Press Start below to begin real-time emotion detection stream.")

                def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
                    img = frame.to_ndarray(format="bgr24")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, face_scale, min_neighbors)
                    dominant = None

                    for (x, y, w, h) in faces:
                        y1, y2 = max(0, y - 20), min(img.shape[0], y + h + 20)
                        x1, x2 = max(0, x - 20), min(img.shape[1], x + w + 20)
                        crop = img[y1:y2, x1:x2]
                        if crop.size > 0:
                            emotion, conf, probs = predict(emotion_model, preprocess_face(crop, model_name))
                            if dominant is None or conf > dominant["conf"]:
                                dominant = {"emotion": emotion, "conf": conf, "probs": probs}

                            color = COLOR_MAP.get(emotion, (148, 163, 184))
                            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

                            label = f"{emotion} {conf:.0%}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
                            cv2.rectangle(img, (x, max(0, y - th - 10)), (x + tw + 10, y), (17, 20, 29), -1)
                            cv2.rectangle(img, (x, max(0, y - th - 10)), (x + tw + 10, y), color, 1)
                            cv2.putText(img, label, (x + 5, max(th + 2, y - 4)), font, 0.6, (240, 245, 255), 1, cv2.LINE_AA)

                    if dominant:
                        hud_text = f"{dominant['emotion']}  {dominant['conf']:.1%}"
                        c = COLOR_MAP.get(dominant["emotion"], (148, 163, 184))
                        cv2.rectangle(img, (16, 16), (250, 52), (17, 20, 29), -1)
                        cv2.rectangle(img, (16, 16), (250, 52), c, 1)
                        cv2.putText(img, hud_text, (26, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 245, 255), 1, cv2.LINE_AA)

                    return av.VideoFrame.from_ndarray(img, format="bgr24")

                webrtc_streamer(
                    key="browser-live-cam",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )

            with col_info:
                st.markdown(clean_html(f"""
                <div class="panel-card">
                    <div class="panel-row"><span>Active Model</span><span>{model_name}</span></div>
                    <div class="panel-row"><span>Hardware Device</span><span>{DEVICE.type.upper()}</span></div>
                    <div class="panel-row"><span>Target Classes</span><span>7 Emotions</span></div>
                </div>
                """), unsafe_allow_html=True)

                st.caption("Target Emotion Classes")
                cat_pills = []
                for cls in CLASSES:
                    c = COLOR_MAP[cls]
                    hex_c = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
                    cat_pills.append(
                        f'<div class="category-pill">'
                        f'<div class="category-dot" style="background-color: {hex_c};"></div>'
                        f'<span>{cls}</span>'
                        f'</div>'
                    )
                cat_joined = "".join(cat_pills)
                st.markdown(f'<div class="category-list">{cat_joined}</div>', unsafe_allow_html=True)

    elif cam_mode == "Browser Snapshot":
        col_cam, col_res = st.columns([1.5, 1])
        with col_cam:
            img_file = st.camera_input("Take photo", label_visibility="collapsed")

        with col_res:
            if img_file:
                img = np.array(Image.open(img_file).convert("RGB"))
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                faces = face_detector.detectMultiScale(gray, face_scale, min_neighbors)
                if len(faces) > 0:
                    results = []
                    for (x, y, w, h) in faces:
                        crop = img[max(0, y - 20):y + h + 20, max(0, x - 20):x + w + 20]
                        emotion, conf, probs = predict(emotion_model, preprocess_face(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), model_name))
                        results.append({"emotion": emotion, "conf": conf, "probs": probs})
                    top_res = max(results, key=lambda x: x["conf"])
                    render_emotion_stats(top_res["emotion"], top_res["conf"], top_res["probs"])
                else:
                    st.markdown(clean_html("""
                    <div class="empty-card">
                        <div class="empty-title">No Face Detected</div>
                        <p class="empty-caption">Position your face clearly in frame or adjust detector sensitivity in the sidebar.</p>
                    </div>
                    """), unsafe_allow_html=True)
            else:
                st.markdown(clean_html("""
                <div class="empty-card">
                    <div class="empty-title">Awaiting Snapshot</div>
                    <p class="empty-caption">Take a photo using the camera viewfinder on the left.</p>
                </div>
                """), unsafe_allow_html=True)

    elif cam_mode == "Host Webcam":
        col_cam, col_res = st.columns([1.5, 1])
        with col_cam:
            run = st.toggle("Start Camera Feed", value=False)
            frame_holder = st.empty()

        with col_res:
            result_holder = st.empty()
            if not run:
                result_holder.markdown(clean_html("""
                <div class="empty-card">
                    <div class="empty-title">Camera Inactive</div>
                    <p class="empty-caption">Toggle 'Start Camera Feed' to begin continuous host webcam inference.</p>
                </div>
                """), unsafe_allow_html=True)

        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not access host camera device. Please ensure webcam is connected.")
            else:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, face_scale, min_neighbors)

                    top_res = None
                    for (x, y, w, h) in faces:
                        crop = frame[max(0, y - 20):y + h + 20, max(0, x - 20):x + w + 20]
                        if crop.size > 0:
                            emotion, conf, probs = predict(emotion_model, preprocess_face(crop, model_name))
                            top_res = {"emotion": emotion, "conf": conf, "probs": probs}
                            color = COLOR_MAP.get(emotion, (148, 163, 184))
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            label = f"{emotion} {conf:.0%}"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
                            cv2.rectangle(frame, (x, max(0, y - th - 10)), (x + tw + 10, y), (17, 20, 29), -1)
                            cv2.rectangle(frame, (x, max(0, y - th - 10)), (x + tw + 10, y), color, 1)
                            cv2.putText(frame, label, (x + 5, max(th + 2, y - 4)), font, 0.6, (240, 245, 255), 1, cv2.LINE_AA)

                    frame_holder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    if top_res:
                        render_emotion_stats(top_res["emotion"], top_res["conf"], top_res["probs"], result_holder)
                    else:
                        result_holder.markdown(clean_html("""
                        <div class="empty-card">
                            <div class="empty-title">Scanning for Faces...</div>
                            <p class="empty-caption">Position your face inside the camera view.</p>
                        </div>
                        """), unsafe_allow_html=True)
                    time.sleep(0.01)
                cap.release()

with tab_upload:
    f = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if f:
        img = np.array(Image.open(f).convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = face_detector.detectMultiScale(gray, face_scale, min_neighbors)
        col_img, col_res = st.columns([1.5, 1])

        with col_img:
            annotated = img.copy()
            for (x, y, w, h) in faces:
                crop = img[max(0, y - 20):y + h + 20, max(0, x - 20):x + w + 20]
                emotion, conf, probs = predict(emotion_model, preprocess_face(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), model_name))
                color = COLOR_MAP.get(emotion, (148, 163, 184))
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                label = f"{emotion} {conf:.0%}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
                cv2.rectangle(annotated, (x, max(0, y - th - 10)), (x + tw + 10, y), (17, 20, 29), -1)
                cv2.rectangle(annotated, (x, max(0, y - th - 10)), (x + tw + 10, y), color, 1)
                cv2.putText(annotated, label, (x + 5, max(th + 2, y - 4)), font, 0.6, (240, 245, 255), 1, cv2.LINE_AA)

            st.image(annotated, use_container_width=True)

        with col_res:
            if len(faces) > 0:
                results = []
                for (x, y, w, h) in faces:
                    crop = img[max(0, y - 20):y + h + 20, max(0, x - 20):x + w + 20]
                    emotion, conf, probs = predict(emotion_model, preprocess_face(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), model_name))
                    results.append({"emotion": emotion, "conf": conf, "probs": probs})
                top_res = max(results, key=lambda x: x["conf"])
                render_emotion_stats(top_res["emotion"], top_res["conf"], top_res["probs"])
            else:
                st.markdown(clean_html("""
                <div class="empty-card">
                    <div class="empty-title">No Face Detected</div>
                    <p class="empty-caption">Try adjusting the 'Scale Factor' or 'Min Neighbors' in the sidebar.</p>
                </div>
                """), unsafe_allow_html=True)
    else:
        st.markdown(clean_html("""
        <div class="empty-card" style="margin-top: 1rem;">
            <div class="empty-title">Upload a Portrait or Face Photo</div>
            <p class="empty-caption">Supports JPG, PNG, and JPEG files.</p>
        </div>
        """), unsafe_allow_html=True)
