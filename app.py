import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import tempfile
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. PAGE CONFIGURATION & CUSTOM CSS ---
st.set_page_config(
    page_title="Sistem Absensi Liveness",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konfigurasi STUN Server (Agar tembus jaringan)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# CSS: Tampilan Bersih & Profesional (Sama Persis dengan Requestmu)
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        
        /* Kartu Statistik Gelap & Elegan */
        .metric-card {
            background-color: #1E1E1E;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            text-align: center;
        }
        
        /* Label Status yang Jelas */
        .status-badge {
            display: inline-block;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 16px;
            text-align: center;
            width: 100%;
            margin-bottom: 15px;
            letter-spacing: 0.5px;
        }
        .badge-safe {
            background-color: #d1e7dd;
            color: #0f5132;
            border: 1px solid #badbcc;
        }
        .badge-danger {
            background-color: #f8d7da;
            color: #842029;
            border: 1px solid #f5c2c7;
        }
        .badge-idle {
            background-color: #e2e3e5;
            color: #41464b;
            border: 1px solid #d3d6d8;
        }
        
        /* Penyesuaian Tab */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            background-color: #0E1117;
            border: 1px solid #333;
            border-radius: 4px;
            padding-top: 8px;
            padding-bottom: 8px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #262730;
            border-bottom: 2px solid #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. BACKEND LOGIC (Shared) ---

@st.cache_resource
def load_model():
    return YOLO("best.pt")

def get_overlay_mask(h, w):
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    center = (int(w/2), int(h/2))
    axes = (int(w*0.22), int(h*0.38)) 
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
    mask_inv = cv2.bitwise_not(mask)
    return mask, mask_inv, center, axes

def draw_dashed_ellipse(img, center, axes, color, thickness=2):
    for angle in range(0, 360, 30):
        cv2.ellipse(img, center, axes, 0, angle, angle + 15, color, thickness)

# Logic Proses Frame (Dipakai di Upload & WebRTC)
def process_frame(model, frame, conf_thresh, mask_info=None):
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
                    status_msg = "VERIFIKASI BERHASIL"
                    color = (0, 255, 0)
                elif label in ['spoof_screen', 'spoof_print']:
                    status_type = "danger"
                    status_msg = "ANOMALI TERDETEKSI"
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 0)

                cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)
                
                # Label Box
                label_text = f"{label.replace('_', ' ').upper()}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(img_display, (x1, y1-25), (x1+tw, y1), color, -1)
                cv2.putText(img_display, label_text, (x1, y1-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    if mask_info is not None:
        mask, mask_inv, center, axes = mask_info
        if mask.shape[:2] != (h, w):
             mask, mask_inv, center, axes = get_overlay_mask(h, w)
             
        outside = (cv2.bitwise_and(img_display, mask_inv) * 0.6).astype(np.uint8)
        inside = cv2.bitwise_and(img_display, mask)
        img_display = cv2.add(inside, outside)
        draw_dashed_ellipse(img_display, center, axes, (200, 200, 200), 2)
        
        cv2.putText(img_display, "Posisikan Wajah", (center[0]-90, h-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

    return img_display, status_type, status_msg, max_conf

# --- 3. WEBRTC PROCESSOR (Engine Khusus Cloud) ---
class CloudVideoProcessor:
    def __init__(self):
        self.model = load_model()
        self.mask_cache = None
        # Hardcode threshold 0.675 untuk cloud agar konsisten
        self.conf_thresh = 0.675 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1) # Mirror effect
        h, w = img.shape[:2]
        
        if self.mask_cache is None or self.mask_cache[0].shape[:2] != (h, w):
            self.mask_cache = get_overlay_mask(h, w)
            
        # Proses Frame
        proc_frame, s_type, s_msg, m_conf = process_frame(
            self.model, img, self.conf_thresh, self.mask_cache
        )
        
        # --- HUD (Head-Up Display) ---
        # Karena di WebRTC kita tidak bisa update UI Streamlit (Sidebar/Badge) secara realtime,
        # Kita gambar status bar langsung di atas video.
        if s_type == "safe":
            color_bg = (0, 200, 0) # Hijau
        elif s_type == "danger":
            color_bg = (0, 0, 200) # Merah
        else:
            color_bg = (50, 50, 50) # Abu

        # Gambar Kotak Status di Pojok Kiri Atas Video
        cv2.rectangle(proc_frame, (0, 0), (w, 40), color_bg, -1)
        status_text = f"{s_msg} ({m_conf:.1%})"
        cv2.putText(proc_frame, status_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(proc_frame, format="bgr24")

# --- 4. LOAD MODEL ---
try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Error: Model 'best.pt' tidak ditemukan.")
    st.stop()

# --- 5. SIDEBAR (PENGATURAN) ---
with st.sidebar:
    st.title("⚙️ Pengaturan")
    
    st.subheader("Parameter AI")
    conf_threshold = st.slider(
        "Akurasi Minimum (Threshold)", 
        min_value=0.0, max_value=1.0, value=0.675, step=0.001, format="%.3f",
        help="Default 0.675 adalah nilai F1-Score optimal."
    )
    
    st.divider()
    
    st.subheader("Bantuan")
    with st.expander("Panduan Penggunaan"):
        st.markdown("""
        **Versi Cloud (WebRTC):**
        - Akses kamera dilakukan via Browser.
        - Status deteksi muncul di atas video (HUD).
        
        **Warna Indikator:**
        - 🟢 **Hijau:** Wajah Asli
        - 🔴 **Merah:** Palsu
        """)
    
    st.info("Sistem v1.0 - Cloud Deployment")

# --- 6. MAIN CONTENT ---
st.title("Sistem Verifikasi Liveness")
st.markdown("Dashboard Monitoring & Validasi Biometrik")

# TAB NAVIGATION
tab_cam, tab_img, tab_vid = st.tabs(["🎥 Live Camera", "🖼️ Upload Foto", "📹 Upload Video"])

# === TAB 1: LIVE WEBCAM (VERSI CLOUD/WEBRTC) ===
with tab_cam:
    col_video, col_info = st.columns([3, 1.2])
    
    with col_info:
        st.markdown("#### Informasi Sistem")
        st.info("💡 **Mode Cloud:**\nSistem menggunakan WebRTC untuk streaming langsung dari browser Anda ke server.")
        st.warning("⚠️ **Performa:**\nKecepatan deteksi (FPS) bergantung pada koneksi internet Anda.")

    with col_video:
        # PENGGANTI cv2.VideoCapture UNTUK CLOUD
        webrtc_streamer(
            key="liveness-cloud",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=CloudVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

# === TAB 2: UPLOAD FOTO (SAMA SEPERTI SEBELUMNYA) ===
with tab_img:
    st.markdown("#### 🖼️ Audit File Foto")
    file = st.file_uploader("Pilih File Foto", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    if file:
        bytes_data = np.asarray(bytearray(file.read()), dtype=np.uint8)
        frame = cv2.imdecode(bytes_data, 1)
        
        with st.spinner('Sedang menganalisis...'):
            res_frame, s_type, s_msg, m_conf = process_frame(model, frame, conf_threshold, mask_info=None)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(res_frame, channels="BGR", caption="Hasil Deteksi", use_column_width=True)
        with c2:
            st.markdown(f'<div class="status-badge badge-{s_type}">{s_msg}</div>', unsafe_allow_html=True)
            st.metric("Skor Keaslian", f"{m_conf:.1%}")
            
            _, buffer = cv2.imencode('.jpg', res_frame)
            st.download_button("📥 Simpan Hasil", buffer.tobytes(), f"hasil_{file.name}", "image/jpeg", use_container_width=True)

# === TAB 3: UPLOAD VIDEO (SAMA SEPERTI SEBELUMNYA) ===
with tab_vid:
    st.markdown("#### 📹 Audit Rekaman Video")
    file_vid = st.file_uploader("Pilih File Video", type=['mp4', 'avi'], label_visibility="collapsed")

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
        
        prog_bar = st.progress(0)
        status_txt = st.empty()
        
        cnt = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            res_frame, _, _, _ = process_frame(model, frame, conf_threshold, mask_info=None)
            out.write(res_frame)
            
            cnt += 1
            if cnt % 10 == 0:
                prog_bar.progress(cnt/total)
                status_txt.text(f"Memproses Frame: {cnt}/{total}")

        cap.release()
        out.release()
        prog_bar.empty()
        status_txt.success("✅ Pemrosesan Selesai!")
        
        with open(out_file, 'rb') as f:
            vid_bytes = f.read()
            
        st.download_button(
            label="📥 Download Video Hasil",
            data=vid_bytes,
            file_name=f"hasil_{file_vid.name}",
            mime="video/mp4",
            use_container_width=True
        )
        
        os.unlink(tfile.name)
        os.unlink(out_file)