# Local Binary Pattern Histogram Face Recognition With OpenCV

1. Proyek ini merupakan implementasi pengambilan dataset citra wajah dan evaluasi metode **Local Binary Pattern Histogram (LBPH)** pada berbagai kondisi intensitas pencahayaan (10 lux, 50 lux, 150 lux). Kode terdiri dari dua skrip utama:

- `capture.py` – untuk mengakuisisi dataset wajah (train dan test) melalui kamera.
- `evaluasi.py`– untuk melatih model LBPH, menguji pada kondisi pencahayaan berbeda, dan menghasilkan laporan lengkap (tabel, confusion matrix, grafik).

2. Pastikan file haarcascade_frontalface_default.xml berada di direktori yang sama dengan capture.py.
File ini bisa diunduh dari repositori OpenCV:
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

3. 📁 Struktur Folder yang Diharapkan

Setelah menjalankan `capture.py` dan `evaluasi.py`, struktur folder akan seperti berikut (sesuaikan jika Anda mengubah konfigurasi):

proyek/
├── dataset/ # (diubah dari "Data" menjadi "dataset")
│ ├── train/
│ │ ├── person_1/
│ │ ├── person_2/
│ │ ├── ...
│ └── test/
│ ├── rendah_10lux/
│ │ ├── person_1/
│ │ ├── person_2/
│ │ ├── ...
│ ├── sedang_50lux/
│ │ └── ...
│ └── tinggi_150lux/
│ └── ...
└── output_evaluasi/ # dihasilkan oleh evaluasi.py
├── hasil_evaluasi_lengkap.txt
├── hasil_evaluasi.json
├── grafik_akurasi.png
├── grafik_presisi_recall.png
└── grafik_avg_confidence.png