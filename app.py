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

# STUN servers for WebRTC peer connections.
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
# Emotion labels in model output order.
CLASSES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
# Emoji for each emotion.
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

# Folder holding the model weight files.
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
# GPU if available, else CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Model Definitions ───
def build_resnet50(num_classes=7):
    """Builds a ResNet50 with a custom classifier head."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

def build_efficientnet(num_classes=7):
    """Builds an EfficientNet-B3 with a custom classifier head."""
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
    """Builds a small Vision Transformer classifier."""
    return timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=num_classes)

# Weight filename for each model.
MODEL_FILES = {
    "ResNet50": "rafdb_model_resnet50.pth",
    "EfficientNetB3": "rafdb_model_efficientnetb3.pth",
    "ViT-Small": "rafdb_model_vit-small.pth"
}

# Constructor for each model architecture.
MODEL_BUILDERS = {
    "ResNet50": lambda: build_resnet50(7),
    "EfficientNetB3": lambda: build_efficientnet(7),
    "ViT-Small": lambda: build_vit(7)
}

@st.cache_resource
def load_model(name):
    """Loads cached model weights by name; None if missing."""
    path = os.path.join(MODEL_DIR, MODEL_FILES[name])
    if not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = MODEL_BUILDERS[name]()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(DEVICE).eval()

@st.cache_resource
def load_face_detector():
    """Loads the cached Haar cascade face detector."""
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Resize, tensorize, and ImageNet-normalize a face.
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_face(face_img, model_name=None):
    """Converts a BGR face crop into a normalized model tensor."""
    pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return img_transform(pil).unsqueeze(0).to(DEVICE)

@torch.no_grad()
def predict(model, tensor):
    """Returns top emotion, its confidence, and all class probabilities."""
    probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return CLASSES[idx], probs[idx], probs

def to_hex(rgb):
    """Converts an RGB tuple to a hex colour string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def analyze_faces(bgr_img, faces):
    """Runs inference on every detected face and returns per-face results."""
    results = []
    for (x, y, w, h) in faces:
        y1, y2 = max(0, y - 20), min(bgr_img.shape[0], y + h + 20)
        x1, x2 = max(0, x - 20), min(bgr_img.shape[1], x + w + 20)
        crop = bgr_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        emotion, conf, probs = predict(emotion_model, preprocess_face(crop, model_name))
        results.append({"box": (x, y, w, h), "emotion": emotion, "conf": conf, "probs": probs})
    return results

def draw_face_box(img, box, emotion, conf, bgr=True):
    """Draws a rounded-corner bracket box with a filled label chip."""
    x, y, w, h = box
    rgb = COLOR_MAP.get(emotion, (148, 163, 184))
    color = (rgb[2], rgb[1], rgb[0]) if bgr else rgb
    # Translucent fill
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)
    # Thin outline + thick corner brackets
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)
    k = max(12, w // 6)
    for (cx, cy, dx, dy) in [(x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
        cv2.line(img, (cx, cy), (cx + dx * k, cy), color, 3, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * k), color, 3, cv2.LINE_AA)
    # Label chip
    label = f"{emotion}  {conf:.0%}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
    top = max(0, y - th - 16)
    cv2.rectangle(img, (x, top), (x + tw + 16, y - 4), color, -1)
    cv2.putText(img, label, (x + 8, y - 10), font, 0.55, (12, 14, 20), 1, cv2.LINE_AA)
    return img

# ─── Styles ───
st.markdown(clean_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Figtree:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root {
    --bg: #ECEEF2;
    --panel: #FFFFFF;
    --ink: #14161B;
    --muted: #6A7180;
    --line: #DCE0E7;
    --cobalt: #2433D6;
    --display: 'Bricolage Grotesque', sans-serif;
    --mono: 'IBM Plex Mono', monospace;
}
html, body, [class*="css"], .stApp, p, label, button, input {
    font-family: 'Figtree', -apple-system, sans-serif;
    color: var(--ink);
}
.stApp { background: var(--bg); }
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
.block-container { padding: 0 1.5rem 4rem 1.5rem !important; max-width: 1200px !important; }
@keyframes settle { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes sweep { from { clip-path: inset(0 100% 0 0); } to { clip-path: inset(0 0 0 0); } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
:focus-visible { outline: 2px solid var(--cobalt) !important; outline-offset: 2px; }
/* Top bar */
.st-key-topbar { border-bottom: 1px solid var(--line); padding: 14px 0; }
.st-key-topbar [data-testid="stColumn"]:last-child { display: flex; justify-content: flex-end; }
.topbar { display: flex; align-items: center; gap: 14px; }
.st-key-topbar [data-testid="stHorizontalBlock"] { align-items: center !important; gap: 8px !important; }
.st-key-topbar [data-testid="stColumn"] { display: flex; align-items: center; min-height: 32px; }
.st-key-topbar [data-testid="stColumn"] > div, .st-key-topbar [data-testid="stVerticalBlock"] { gap: 0 !important; width: 100%; }
.st-key-topbar [data-testid="stElementContainer"], .st-key-topbar [data-testid="stMarkdown"],
.st-key-topbar [data-testid="stMarkdownContainer"], .st-key-topbar [data-testid="stPopover"] { margin: 0 !important; padding: 0 !important; }
.st-key-topbar [data-testid="stMarkdownContainer"] > div { margin: 0 !important; }
.st-key-topbar [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"] { align-items: flex-end; }
.wordmark { font-family: var(--display); font-weight: 800; font-size: 26px; letter-spacing: -0.04em; line-height: 1; }
.wordmark span { color: var(--cobalt); }
.tagline { font-size: 13px; color: var(--muted); padding-left: 14px; border-left: 1px solid var(--line); }
.status {
    margin-left: auto; display: inline-flex; align-items: center; gap: 8px;
    font-family: var(--mono); font-size: 11.5px; color: var(--muted);
    background: var(--panel); border: 1px solid var(--line); border-radius: 999px; height: 32px; padding: 0 12px; box-sizing: border-box;
}
.status i { width: 7px; height: 7px; border-radius: 50%; background: #16A34A; display: inline-block; }
.status.off i { background: #DC2626; }
/* Page intro */
.intro { padding: 2.6rem 0 1.8rem 0; }
.intro h1 {
    font-family: var(--display); font-weight: 600; font-size: clamp(1.9rem, 3.6vw, 2.7rem);
    letter-spacing: -0.035em; line-height: 1.05; margin: 0; color: var(--ink);
}
.intro p { color: var(--muted); font-size: 16px; line-height: 1.5; margin: 12px 0 0 0; max-width: 480px; }
/* Settings popover button */
.st-key-topbar [data-testid="stPopover"] { width: auto !important; }
.st-key-topbar [data-testid="stPopover"] button {
    background: var(--panel) !important; border: 1px solid var(--line) !important;
    border-radius: 999px !important; color: var(--ink) !important;
    min-height: 0 !important; height: 32px; padding: 0 12px !important; width: auto !important;
}
.st-key-topbar [data-testid="stPopover"] button p { font-size: 13px; font-weight: 500; }
.st-key-topbar [data-testid="stPopover"] button:hover { border-color: #C3C9D4 !important; }
/* Source switcher */
[data-testid="stButtonGroup"] > div {
    background: #E1E4EA; padding: 4px; border-radius: 12px; gap: 2px !important; width: fit-content;
}
[data-testid="stButtonGroup"] button {
    border: none !important; background: transparent !important; border-radius: 9px !important;
    padding: 6px 18px !important; min-height: 36px; color: var(--muted) !important; box-shadow: none !important;
}
[data-testid="stButtonGroup"] button p { font-size: 14px; font-weight: 500; }
[data-testid="stButtonGroup"] button:hover { color: var(--ink) !important; }
[data-testid="stButtonGroup"] button[kind$="Active"] {
    background: var(--panel) !important; color: var(--ink) !important;
    box-shadow: 0 1px 3px rgba(20,22,27,.10), 0 0 0 1px rgba(20,22,27,.05) !important;
}
.source-hint { font-size: 13.5px; color: var(--muted); text-align: right; }
@media (max-width: 640px) { .source-hint { text-align: left; } }
/* Tabs as a segmented control */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #E1E4EA; padding: 3px; border-radius: 10px; width: fit-content; margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab"] {
    height: 34px; border-radius: 8px; padding: 0 18px; color: var(--muted);
    font-weight: 500; font-size: 13.5px; background: transparent; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--panel) !important; color: var(--ink) !important;
    box-shadow: 0 1px 2px rgba(20,22,27,.08), 0 0 0 1px rgba(20,22,27,.04);
}
.stTabs [data-baseweb="tab-border"], .stTabs [data-baseweb="tab-highlight"] { display: none; }
div[role="radiogroup"] { gap: 18px !important; margin-bottom: .6rem; }
div[role="radiogroup"] label p { font-size: 13.5px; }
/* Media frames */
[data-testid="stImage"] img, [data-testid="stCameraInput"], video {
    border-radius: 14px !important; overflow: hidden; background: var(--panel);
    box-shadow: 0 0 0 1px var(--line);
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel) !important; border: 1.5px dashed #C3C9D4 !important;
    border-radius: 14px !important; padding: 1.6rem !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--cobalt) !important; }
/* Reading panel — the instrument */
.reading {
    background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
    padding: 22px 24px 20px 24px; animation: settle .3s ease-out both;
}
.reading-head {
    display: flex; justify-content: space-between; align-items: center;
    font-family: var(--mono); font-size: 11px; color: var(--muted); letter-spacing: .04em;
}
.reading-word {
    font-family: var(--display); font-weight: 800; font-size: clamp(3rem, 6vw, 4.6rem);
    letter-spacing: -0.05em; line-height: .95; margin: 18px 0 6px 0;
}
.reading-conf { font-family: var(--mono); font-size: 13px; color: var(--muted); }
.reading-conf b { color: var(--ink); font-weight: 500; }
.spectrum {
    display: flex; height: 14px; border-radius: 4px; overflow: hidden; gap: 2px;
    margin: 22px 0 8px 0; animation: sweep .7s cubic-bezier(.2,.8,.2,1) both;
}
.spectrum div { min-width: 2px; }
.spectrum-scale { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 10px; color: #A2A8B4; }
.legend { margin-top: 18px; border-top: 1px solid var(--line); }
.legend-row {
    display: grid; grid-template-columns: 10px 1fr auto; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid #EEF0F3; font-size: 13.5px; color: var(--muted);
}
.legend-row:last-child { border-bottom: none; }
.legend-row i { width: 10px; height: 10px; border-radius: 3px; display: block; }
.legend-row span:last-child { font-family: var(--mono); font-size: 12px; }
.legend-row.top { color: var(--ink); font-weight: 600; }
.legend-row.top span:last-child { color: var(--ink); }
/* Empty / guidance state */
.empty-card {
    background: var(--panel); border: 1px solid var(--line); border-radius: 18px;
    padding: 2.4rem 1.6rem; text-align: center; animation: settle .3s ease-out both;
}
.empty-icon {
    width: 46px; height: 46px; margin: 0 auto 14px auto; display: grid; place-items: center;
    font-size: 20px; border-radius: 12px; background: #F2F4F8; border: 1px solid var(--line);
}
.empty-title { font-family: var(--display); font-weight: 600; font-size: 1.15rem; letter-spacing: -0.02em; margin-bottom: 6px; }
.empty-caption { font-size: 14px; color: var(--muted); margin: 0 auto; line-height: 1.5; max-width: 340px; }
.empty-caption code { background: #F2F4F8; padding: 1px 6px; border-radius: 5px; font-family: var(--mono); font-size: 12px; }
/* Box colour key */
.key-card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; margin-top: 12px; }
.key-title { font-family: var(--display); font-weight: 600; font-size: 15px; letter-spacing: -0.01em; }
.key-sub { font-size: 12.5px; color: var(--muted); margin: 2px 0 12px 0; line-height: 1.45; }
.key-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 16px; }
.key-item { display: flex; align-items: center; gap: 10px; font-size: 13.5px; }
.key-item i {
    width: 14px; height: 14px; flex-shrink: 0; border: 2.5px solid; border-radius: 3px;
}
/* Box colour key end */
/* Info panel + class chips */
.section-label { font-family: var(--mono); font-size: 11px; color: var(--muted); margin: 1.2rem 0 .6rem 0; }
.category-list { display: flex; flex-wrap: wrap; gap: 6px; }
.category-pill {
    display: inline-flex; align-items: center; gap: 7px; padding: 5px 11px 5px 9px;
    background: var(--panel); border: 1px solid var(--line); border-radius: 8px; font-size: 12.5px;
}
.category-dot { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }
.panel-card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 4px 16px; }
.panel-row {
    display: flex; justify-content: space-between; font-size: 13px; padding: 10px 0;
    color: var(--muted); border-bottom: 1px solid #EEF0F3;
}
.panel-row:last-child { border-bottom: none; }
.panel-row span:last-child { color: var(--ink); font-family: var(--mono); font-size: 12px; }
@media (max-width: 640px) { .tagline { display: none; } .block-container { padding: 0 1rem 3rem 1rem !important; } }
</style>
"""), unsafe_allow_html=True)

def empty_state(icon, title, caption, container=None):
    """Renders a placeholder card with icon, title, and caption."""
    html = (
        f'<div class="empty-card"><div class="empty-icon">{icon}</div>'
        f'<div class="empty-title">{title}</div><p class="empty-caption">{caption}</p></div>'
    )
    (container or st).markdown(html, unsafe_allow_html=True)

# ─── Top bar ───
topbar = st.container(key="topbar")
bar_left, bar_right = topbar.columns([8, 1], vertical_alignment="center", gap="small")
with bar_right:
    with st.popover("Settings", icon=":material/tune:"):
        model_name = st.selectbox(
            "Model",
            list(MODEL_FILES.keys()),
            index=0,  # Default to ResNet50
            help="ResNet50 is the most accurate. ViT-Small is the lightest."
        )
        face_scale = st.slider("Detection scale", 1.05, 1.5, 1.3, step=0.05, help="Lower finds smaller or farther faces, but runs slower.")
        min_neighbors = st.slider("Detection strictness", 1, 10, 5, step=1, help="Higher ignores more false matches, but may miss real faces.")
        st.caption("No face found? Lower the scale or strictness. Face the camera in even light.")

face_detector = load_face_detector()
emotion_model = load_model(model_name)

with bar_left:
    status_cls = "" if emotion_model is not None else " off"
    status_txt = f"{model_name} · {DEVICE.type.upper()}" if emotion_model is not None else f"{model_name} weights missing"
    st.markdown(
        f'<div class="topbar"><div class="wordmark">mien<span>.</span></div>'
        f'<div class="tagline">Expression reading</div>'
        f'<div class="status{status_cls}"><i></i>{status_txt}</div></div>',
        unsafe_allow_html=True
    )

def render_emotion_stats(emotion, conf, probs, container=None, n_faces=1):
    """Renders the reading panel with spectrum bar and probability legend."""
    dom_hex = to_hex(COLOR_MAP.get(emotion, (148, 163, 184)))
    order = np.argsort(probs)[::-1]

    bands = "".join(
        f'<div title="{CLASSES[i]} {probs[i]:.1%}" style="flex: {max(probs[i], 0.004):.4f}; background: {to_hex(COLOR_MAP[CLASSES[i]])};"></div>'
        for i in order
    )
    rows = "".join(
        f'<div class="legend-row{" top" if CLASSES[i] == emotion else ""}">'
        f'<i style="background: {to_hex(COLOR_MAP[CLASSES[i]])};"></i>'
        f'<span>{CLASSES[i]}</span><span>{probs[i]:.1%}</span></div>'
        for i in order
    )
    faces_txt = f"{n_faces} face{'s' if n_faces != 1 else ''} · strongest shown"
    html = (
        f'<div class="reading">'
        f'<div class="reading-head"><span>Reading</span><span>{faces_txt}</span></div>'
        f'<div class="reading-word" style="text-decoration: underline; text-decoration-color: {dom_hex}; text-decoration-thickness: 7px; text-underline-offset: 10px;">{emotion}</div>'
        f'<div class="reading-conf"><b>{conf:.0%}</b> confidence</div>'
        f'<div class="spectrum">{bands}</div>'
        f'<div class="spectrum-scale"><span>0</span><span>Expression spectrum</span><span>100%</span></div>'
        f'<div class="legend">{rows}</div>'
        f'</div>'
    )
    (container or st).markdown(html, unsafe_allow_html=True)

def render_top_result(results, container=None):
    """Shows the stats for the most confident detected face."""
    top = max(results, key=lambda r: r["conf"])
    render_emotion_stats(top["emotion"], top["conf"], top["probs"], container, n_faces=len(results))

st.markdown(clean_html("""
<div class="intro">
    <h1>See what a face is saying.</h1>
    <p>Point a camera or drop in a photo. Mien finds each face and reads its expression across seven emotions.</p>
</div>
"""), unsafe_allow_html=True)

# Helper text for each input source.
SOURCE_HINTS = {
    "Live": "Streams from this browser's camera and reads every frame.",
    "Snapshot": "Takes one photo with this browser's camera.",
    "Upload": "Reads a JPG or PNG from your files.",
    "Local webcam": "Uses the camera attached to the machine running Mien.",
}
src_col, hint_col = st.columns([1.15, 1], vertical_alignment="center")
with src_col:
    cam_mode = st.segmented_control(
        "Source", list(SOURCE_HINTS), default="Live", label_visibility="collapsed"
    ) or "Live"
with hint_col:
    st.markdown(f'<div class="source-hint">{SOURCE_HINTS[cam_mode]}</div>', unsafe_allow_html=True)

if cam_mode != "Upload":
    if cam_mode == "Live":
        if not WEBRTC_AVAILABLE:
            empty_state(
                "📡", "Live mode needs streamlit-webrtc",
                "Use <strong>Snapshot</strong> instead, "
                "or run <code>pip install streamlit-webrtc av</code> for continuous streaming."
            )
        else:
            col_cam, col_info = st.columns([1.6, 1], gap="large")
            with col_cam:
                st.caption("Press Start to read expressions continuously.")

                def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
                    """Detects and annotates faces on each live video frame."""
                    img = frame.to_ndarray(format="bgr24")
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, face_scale, min_neighbors)
                    for r in analyze_faces(img, faces):
                        draw_face_box(img, r["box"], r["emotion"], r["conf"])
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
                    <div class="panel-row"><span>Model</span><span>{model_name}</span></div>
                    <div class="panel-row"><span>Device</span><span>{DEVICE.type.upper()}</span></div>
                    <div class="panel-row"><span>Weights</span><span>{MODEL_FILES[model_name]}</span></div>
                    <div class="panel-row"><span>Classes</span><span>7</span></div>
                </div>
                """), unsafe_allow_html=True)

                keys = "".join(
                    f'<div class="key-item"><i style="border-color: {to_hex(COLOR_MAP[cls])};"></i>{cls}</div>'
                    for cls in CLASSES
                )
                st.markdown(
                    f'<div class="key-card"><div class="key-title">Box colours</div>'
                    f'<p class="key-sub">Each face in the stream is framed in the colour of its expression.</p>'
                    f'<div class="key-grid">{keys}</div></div>',
                    unsafe_allow_html=True
                )

    elif cam_mode == "Snapshot":
        col_cam, col_res = st.columns([1.5, 1], gap="large")
        with col_cam:
            img_file = st.camera_input("Take photo", label_visibility="collapsed")

        with col_res:
            if img_file:
                bgr = cv2.cvtColor(np.array(Image.open(img_file).convert("RGB")), cv2.COLOR_RGB2BGR)
                faces = face_detector.detectMultiScale(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), face_scale, min_neighbors)
                results = analyze_faces(bgr, faces)
                if results:
                    render_top_result(results)
                else:
                    empty_state("🔍", "No face detected", "Position your face clearly in frame or lower detection strictness in Settings.")
            else:
                empty_state("📸", "Take a snapshot", "Use the camera on the left. The reading appears here.")

    elif cam_mode == "Local webcam":
        col_cam, col_res = st.columns([1.5, 1], gap="large")
        with col_cam:
            run = st.toggle("Start webcam", value=False)
            frame_holder = st.empty()

        with col_res:
            result_holder = st.empty()
            if not run:
                empty_state("🎥", "Webcam is off", "Turn on <strong>Start webcam</strong> to read expressions continuously.", result_holder)

        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Webcam not found. Connect a camera, or close other apps using it, then turn it on again.")
            else:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    faces = face_detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), face_scale, min_neighbors)
                    results = analyze_faces(frame, faces)
                    for r in results:
                        draw_face_box(frame, r["box"], r["emotion"], r["conf"])

                    frame_holder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    if results:
                        render_top_result(results, result_holder)
                    else:
                        empty_state("👀", "Scanning for faces…", "Position your face inside the camera view.", result_holder)
                    time.sleep(0.01)
                cap.release()

if cam_mode == "Upload":
    f = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    if f:
        bgr = cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = face_detector.detectMultiScale(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), face_scale, min_neighbors)
        results = analyze_faces(bgr, faces)
        col_img, col_res = st.columns([1.5, 1], gap="large")

        with col_img:
            annotated = bgr.copy()
            for r in results:
                draw_face_box(annotated, r["box"], r["emotion"], r["conf"])
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col_res:
            if results:
                render_top_result(results)
            else:
                empty_state("🔍", "No face detected", "Try lowering <strong>Detection scale</strong> or <strong>Detection strictness</strong> in Settings.")
    else:
        empty_state("🖼️", "Add a photo to read", "Drag a JPG or PNG above. Every face in it gets read; the strongest reading is shown.")

