import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
import gdown
from mtcnn.mtcnn import MTCNN
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Deepfake", page_icon="🧠", layout="centered")

# Fungsi untuk mengunduh model dari Google Drive
@st.cache_resource
def load_model():
    file_id = '1jgfFQrm0HkjDbGOpU6rb_7rEfZ1xzVZC'  
    model_path = 'EfficientNetB4.h5'

    if not os.path.exists(model_path):
        with st.spinner("🔄 Mengunduh model dari Google Drive..."):
            gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)

    return tf.keras.models.load_model(model_path)

# Muat model dan detektor wajah
model = load_model()

@st.cache_resource
def load_detector():
    return MTCNN()

detector = load_detector()

# Fungsi untuk ekstraksi wajah
def extract_face(image_np, padding=20, target_size=380):
    faces = detector.detect_faces(image_np)
    if not faces:
        return None
    face = sorted(faces, key=lambda x: x['confidence'], reverse=True)[0]
    x, y, w, h = face['box']
    x, y = max(0, x - padding), max(0, y - padding)
    w, h = w + 2 * padding, h + 2 * padding
    h_img, w_img, _ = image_np.shape
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    face_crop = image_np[y:y+h, x:x+w]
    resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return resized

# Navigasi sidebar
st.sidebar.title("🔍 Navigasi")
menu = st.sidebar.radio("📂 Menu", ["🏠 Beranda", "🧠 Deteksi Deepfake"])

# Halaman Beranda
if menu == "🏠 Beranda":
    st.markdown("<h1 style='text-align: center;'>📌 Deteksi Gambar Deepfake</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: justify; font-size: 17px;">
        Selamat datang di aplikasi <b>Deteksi Gambar Deepfake</b> berbasis <b>EfficientNet-B4</b> dan <b>Streamlit</b>!<br><br>

        Aplikasi ini mendeteksi gambar wajah yang telah dimanipulasi menggunakan teknik <b>deepfake</b> dengan tahapan:
        <ul>
        <li>📤 Mengunggah gambar wajah</li>
        <li>🔍 Deteksi wajah dengan <b>MTCNN</b></li>
        <li>✂️ Crop & resize wajah menjadi 380x380 piksel</li>
        <li>🤖 Prediksi menggunakan model <b>EfficientNet-B4</b></li>
        </ul>

        Hasil prediksi akan menunjukkan apakah gambar termasuk <b>REAL (Asli)</b> atau <b>FAKE (Palsu)</b> disertai probabilitas keyakinan model.
        <br><br>
        👉 Klik menu <b>🧠 Deteksi Deepfake</b> di sebelah kiri untuk memulai.
        </div>
        """,
        unsafe_allow_html=True
    )

# Halaman Deteksi Deepfake
elif menu == "🧠 Deteksi Deepfake":
    st.markdown("<h1 style='text-align: center;'>🧠 Deteksi Gambar Deepfake</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Unggah gambar wajah (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.markdown("### 📸 Gambar yang Anda unggah:")
        st.image(image_np, use_column_width=True)

        st.markdown("### 🔍 Mendeteksi wajah...")
        face_img = extract_face(image_np)

        if face_img is not None:
            st.success("✅ Wajah berhasil terdeteksi.")
            st.markdown("### ✂️ Hasil crop dan resize:")
            st.image(face_img, width=300)

            st.markdown("### 🤖 Melakukan prediksi...")
            input_array = preprocess_input(face_img.astype(np.float32))
            input_array = np.expand_dims(input_array, axis=0)

            prediction = model.predict(input_array)[0][0]
            label = "🟢 REAL (Asli)" if prediction >= 0.5 else "🔴 FAKE (Palsu)"
            confidence = prediction if label.startswith("🟢") else 1 - prediction

            st.markdown(f"<h2 style='color:#117A65;'>Hasil Deteksi: {label}</h2>", unsafe_allow_html=True)
            st.markdown(f"<h4>🔢 Probabilitas: <b>{confidence * 100:.2f}%</b></h4>", unsafe_allow_html=True)
            st.caption("Prediksi berdasarkan ambang batas 0.5")
        else:
            st.error("🚫 Wajah tidak terdeteksi. Pastikan wajah menghadap kamera dengan jelas.")
