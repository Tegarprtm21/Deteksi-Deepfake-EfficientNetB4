# 🧠 Deteksi Gambar Deepfake dengan EfficientNet-B4

Aplikasi ini merupakan sistem deteksi gambar wajah palsu (**deepfake**) berbasis *deep learning* yang dikembangkan menggunakan model **EfficientNet-B4** dan dijalankan melalui antarmuka interaktif berbasis **Streamlit**.

## 📌 Deskripsi

**Deepfake** adalah teknik manipulasi wajah menggunakan kecerdasan buatan yang berpotensi disalahgunakan untuk menyebarkan informasi palsu. Aplikasi ini dirancang untuk membantu mendeteksi apakah gambar wajah merupakan hasil manipulasi (palsu) atau asli.

### ⚙️ Alur Proses Deteksi:

1. 📤 Unggah gambar wajah (format `.jpg`, `.jpeg`, atau `.png`)
2. 🔍 Deteksi wajah menggunakan **MTCNN**
3. ✂️ Potong dan ubah ukuran gambar wajah ke 380x380 piksel
4. 🧠 Prediksi hasil menggunakan model **EfficientNet-B4**
5. 📊 Tampilkan hasil klasifikasi: **REAL (Asli)** atau **FAKE (Palsu)** lengkap dengan probabilitas keyakinan model

## 🚀 Demo Aplikasi

🔗 [Klik untuk mencoba aplikasi](https://deteksi-deepfake-efficientnetb4.streamlit.app/)

## 🛠️ Cara Menjalankan Secara Lokal

1. Clone repository ini:

   ```bash
   git clone https://github.com/Tegarprtm21/Deepfake_EfficientNetB4.git
   cd Deepfake_EfficientNetB4
   ```

2. Install semua dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi Streamlit:

   ```bash
   streamlit run app.py
   ```

> **Catatan:** Pastikan file model `EfficientNetB4.h5` tersedia di dalam folder `model/`. Jika belum ada, aplikasi akan mengunduh model secara otomatis dari Google Drive.

## 📦 Dependencies

Berikut adalah library yang digunakan dalam proyek ini:

* `tensorflow`
* `streamlit`
* `numpy<2.0`
* `opencv-python-headless==4.7.0.72`
* `mtcnn`
* `Pillow`
* `gdown`

(Dependencies sudah disediakan di file `requirements.txt`)

## 👨‍💻 Pengembang

**Tegar Pratama**
Mahasiswa Teknik Informatika
Universitas Nusa Putra

## 📄 Lisensi

Proyek ini dibuat untuk tujuan riset dan pembelajaran. Tidak diperkenankan untuk penggunaan komersial tanpa izin.
