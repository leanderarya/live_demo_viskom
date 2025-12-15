import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. CONFIG & CSS ---
st.set_page_config(
    page_title="Sistem Absensi Liveness",
    page_icon="🛡️",
    layout="wide"
)

# Konfigurasi STUN Server
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# CSS Custom
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        
        /* Badges */
        .status-badge {
            padding: 8px 16px; border-radius: 50px; font-weight: 700;
            text-align: center; width: 100%; margin-bottom: 10px;
        }
        .badge-safe { background-color: #d1e7dd; color: #0f5132; border: 1px solid #badbcc; }
        .badge-danger { background-color: #f8d7da; color: #842029; border: 1px solid #f5c2c7; }
        
        /* Centering WebRTC Component if needed */
        div[data-testid="stVerticalBlock"] > div > div > div > div[data-testid="stWebrtcStreamer"] {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. BACKEND LOGIC ---

@st.cache_resource
def load_model():
    return YOLO("best.pt")

def get_overlay_mask(h, w):
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    center = (int(w/2), int(h/2))
    
    # --- LOGIKA ANTI-GEPENG (Fixed Aspect Ratio) ---
    min_dim = min(w, h)
    base_radius = int(min_dim * 0.35) # Perkecil sedikit jadi 35%
    
    # Paksa rasio 3:4 (Wajah)
    radius_y = int(base_radius * 1.2)
    radius_x = int(base_radius * 0.9)
    
    # Safety bounds
    radius_x = min(radius_x, int(w/2) - 10)
    radius_y = min(radius_y, int(h/2) - 10)
    
    axes = (radius_x, radius_y)
    # -----------------------------------------------
    
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
    mask_inv = cv2.bitwise_not(mask)
    return mask, mask_inv, center, axes

def draw_dashed_ellipse(img, center, axes, color, thickness=2):
    for angle in range(0, 360, 30):
        cv2.ellipse(img, center, axes, 0, angle, angle + 15, color, thickness)

# Fungsi Process Frame
def process_frame_logic(model, frame, conf_thresh, mask_info=None):
    img_display = frame.copy()
    h, w, _ = frame.shape

    results = model(frame, verbose=False, conf=conf_thresh)
    
    status_type = "idle"
    status_msg = "Menunggu..."
    max_conf = 0.0
    
    if results:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                
                if conf > max_conf: max_conf = conf

                if label == 'wajah_asli':
                    status_type = "safe"
                    status_msg = "AMAN"
                    color = (0, 255, 0)
                elif label in ['spoof_screen', 'spoof_print']:
                    status_type = "danger"
                    status_msg = "PALSU"
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 0)

                cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                
                # Label box
                label_text = f"{label.replace('_', ' ').upper()}"
                cv2.putText(img_display, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Overlay
    if mask_info is not None:
        mask, mask_inv, center, axes = mask_info
        if mask.shape[:2] != (h, w):
             mask, mask_inv, center, axes = get_overlay_mask(h, w)
             
        outside = (cv2.bitwise_and(img_display, mask_inv) * 0.6).astype(np.uint8)
        inside = cv2.bitwise_and(img_display, mask)
        img_display = cv2.add(inside, outside)
        draw_dashed_ellipse(img_display, center, axes, (200, 200, 200), 2)
        cv2.putText(img_display, "Posisikan Wajah", (center[0]-80, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return img_display, status_type, status_msg, max_conf

# --- 3. WEBRTC PROCESSOR ---
class CloudVideoProcessor:
    def __init__(self):
        self.model = load_model()
        self.mask_cache = None
        self.conf_thresh = 0.675 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror effect
        h, w = img.shape[:2]
        
        if self.mask_cache is None or self.mask_cache[0].shape[:2] != (h, w):
            self.mask_cache = get_overlay_mask(h, w)
            
        proc_frame, s_type, s_msg, m_conf = process_frame_logic(
            self.model, img, self.conf_thresh, self.mask_cache
        )
        
        # HUD Status
        if s_type == "safe": color_bg = (0, 200, 0)
        elif s_type == "danger": color_bg = (0, 0, 200)
        else: color_bg = (50, 50, 50)
            
        cv2.rectangle(proc_frame, (0, 0), (w, 40), color_bg, -1)
        status_text = f"STATUS: {s_msg} ({m_conf:.1%})"
        cv2.putText(proc_frame, status_text, (20, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(proc_frame, format="bgr24")

# --- 4. MAIN UI ---
try:
    model = load_model()
except:
    st.error("Model best.pt tidak ditemukan.")
    st.stop()

st.title("Sistem Verifikasi Liveness")
st.caption("Dashboard Monitoring & Validasi Biometrik (Cloud Version)")

tab_cam, tab_img, tab_vid = st.tabs(["🎥 Live Camera", "🖼️ Upload Foto", "📹 Upload Video"])

# === TAB 1: LIVE CAM (Compact Layout) ===
with tab_cam:
    # LAYOUT: Kolom Kiri (Video) - Kolom Kanan (Info)
    # Rasio [1.5, 1] agar video tidak terlalu lebar (dominan tapi compact)
    col1, col2 = st.columns([1.5, 1])
    
    with col2:
        st.markdown("#### Info Status")
        st.info("💡 **Akses Kamera:**\nKlik 'SELECT DEVICE' atau 'START' di player video.")
        st.warning("Pastikan wajah berada di dalam oval panduan.")

    with col1:
        # WebRTC Streamer
        # Video akan mengikuti lebar kolom 'col1' yang sudah kita persempit (Rasio 1.5)
        webrtc_streamer(
            key="liveness-cloud",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=CloudVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

# === TAB 2: UPLOAD FOTO (Compact Layout) ===
with tab_img:
    st.markdown("#### 🖼️ Audit File Foto")
    file = st.file_uploader("Pilih File", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    if file:
        bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
        frame = cv2.imdecode(bytes_data, 1)
        res_frame, s_type, s_msg, m_conf = process_frame_logic(model, frame, 0.675, None)

        # LAYOUT: Kolom kiri (Gambar) - Kolom kanan (Status)
        # Spacer di awal agar tidak mepet kiri
        c1, c2 = st.columns([2, 1]) 
        
        with c1:
            # PENTING: width=500 membatasi ukuran gambar agar tidak fullscreen di laptop
            st.image(res_frame, channels="BGR", width=500, caption="Hasil Analisis")
            
        with c2:
            st.markdown(f'<div class="status-badge badge-{s_type}">{s_msg}</div>', unsafe_allow_html=True)
            st.metric("Skor Keaslian", f"{m_conf:.1%}")
            
            _, buffer = cv2.imencode('.jpg', res_frame)
            st.download_button("📥 Download Hasil", buffer.tobytes(), f"hasil_{file.name}", "image/jpeg", use_container_width=True)

# === TAB 3: UPLOAD VIDEO (Compact Layout) ===
with tab_vid:
    st.markdown("#### 📹 Audit Rekaman Video")
    file_vid = st.file_uploader("Pilih Video", type=['mp4', 'avi'], label_visibility="collapsed")

    if file_vid:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(file_vid.read())
        
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps_in, (width, height))
        
        # Layout Progress Bar di tengah, tidak terlalu lebar
        c_prog, _ = st.columns([2, 1])
        with c_prog:
            prog_bar = st.progress(0)
            status_txt = st.empty()
        
        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res_frame, _, _, _ = process_frame_logic(model, frame, 0.675, None)
            out.write(res_frame)
            cnt += 1
            if cnt % 10 == 0: 
                prog_bar.progress(cnt/total)
                status_txt.text(f"Processing: {int((cnt/total)*100)}%")

        cap.release(); out.release(); prog_bar.empty(); status_txt.empty()
        
        with open(out_file, 'rb') as f: vid_bytes = f.read()
        
        # Tombol Download di kolom kiri agar sejajar
        c_dl, _ = st.columns([2, 1])
        with c_dl:
            st.success("Selesai!")
            st.download_button("📥 Download Video Hasil", vid_bytes, f"res_{file_vid.name}", "video/mp4", use_container_width=True)
            
        os.unlink(tfile.name); os.unlink(out_file)