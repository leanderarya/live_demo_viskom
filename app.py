import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. CONFIG ---
st.set_page_config(page_title="E-Presensi Cloud", page_icon="☁️", layout="wide")

# Konfigurasi STUN Server (Agar bisa tembus firewall jaringan kampus/umum)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- 2. HELPERS (Optimized) ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

def draw_overlay(img):
    h, w = img.shape[:2]
    # Overlay statis sederhana agar ringan di cloud
    center = (int(w/2), int(h/2))
    axes = (130, 180)
    
    # Gambar Oval Dashed (Manual)
    for angle in range(0, 360, 30):
        cv2.ellipse(img, center, axes, 0, angle, angle + 15, (200, 200, 200), 2)
    
    # Teks
    cv2.putText(img, "POSISIKAN WAJAH", (center[0]-90, h-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return img

# --- 3. CALLBACK PROCESSOR (Jantung WebRTC) ---
class VideoProcessor:
    def __init__(self):
        self.model = load_model()
        self.skip_counter = 0
        self.last_results = None

    def recv(self, frame):
        # Konversi frame WebRTC (av.VideoFrame) ke OpenCV (numpy)
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror effect
        img = cv2.flip(img, 1)

        # --- LOGIKA YOLO (Frame Skipping untuk Cloud) ---
        # Kita skip lebih agresif (tiap 5 frame) karena CPU Cloud lambat
        if self.skip_counter % 5 == 0:
            # Resize agar ringan diproses model (opsional, misal ke 320sz)
            # results = self.model(img, conf=0.6, imgsz=320, verbose=False)
            results = self.model(img, conf=0.6, verbose=False)
            self.last_results = results
        else:
            results = self.last_results
        
        self.skip_counter += 1

        # --- GAMBAR KOTAK ---
        if results:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label = result.names[cls]

                    if label == 'wajah_asli':
                        color = (0, 255, 0)
                        txt = "AMAN"
                    else:
                        color = (0, 0, 255)
                        txt = "PALSU"
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, txt, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # --- OVERLAY ---
        img = draw_overlay(img)

        # Kembalikan frame ke browser (convert balik ke av.VideoFrame)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. UI STREAMLIT ---
st.title("E-Presensi Cloud ☁️")
st.caption("Powered by WebRTC & YOLO")

col1, col2 = st.columns([3, 1])

with col1:
    # Component WebRTC Pengganti cv2.VideoCapture
    webrtc_streamer(
        key="liveness-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.info("Status: Siap")
    st.markdown("""
    **Panduan:**
    1. Izinkan akses kamera browser.
    2. Tunggu koneksi stabil (State: Playing).
    3. Jika lag, itu wajar karena proses via Cloud.
    """)