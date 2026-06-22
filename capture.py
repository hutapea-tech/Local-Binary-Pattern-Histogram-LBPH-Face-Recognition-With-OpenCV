"""
Kode capture.py — Pengambilan Dataset Wajah (Train & Test)
=====================================================================
Topik Skripsi : Analisis Pengaruh Intensitas Pencahayaan Terhadap Akurasi LBPH
Penulis       : Adi Saputra Hutavea

Petunjuk Penggunaan:
1. Jalankan skrip ini lewat terminal/command prompt.
2. Pilih Mode (Train 45 citra atau Test 15 citra per kondisi).
3. Pilih Subjek (Person 1 - 5).
4. Posisikan wajah di depan kamera, pastikan lux meter sesuai target intensitas cahaya.
5. Proses capture otomatis berjalan dengan jeda (delay) yang ditentukan.
"""

import cv2
import os
import time

# ================= CONFIG =================
BASE_PATH     = "Data"  # Jika ingin langsung dibaca oleh evaluasi.py, ganti menjadi "dataset"
IMG_SIZE      = (100, 100)
MIN_FACE_SIZE = 150

# Jeda antar pengambilan gambar dalam detik.
# Nilai 0.2 memastikan variasi pose/ekspresi mikro dan pengambilan yang cepat.
CAPTURE_DELAY = 0.2

# Jumlah frame warm-up yang dibuang sebelum mulai capture.
# 60 frame ~ 2 detik pada kamera 30fps, cukup untuk stabilisasi sensor dan ring light.
WARMUP_FRAMES = 60

# Load Cascade Classifier untuk Deteksi Wajah
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("[ERROR] File 'haarcascade_frontalface_default.xml' tidak ditemukan!")
    print("Pastikan file tersebut berada di folder yang sama dengan skrip ini.")
    exit(1)

# ================= KONDISI DATASET =================
NAMA_PERSON = [f"person_{i}" for i in range(1, 6)]
KONDISI_MENU = {
    "1": {"nama": "Train (Normal 300 Lux)", "sub_folder": "train", "target": 45},
    "2": {"nama": "Test - Rendah 10 Lux",  "sub_folder": "test/rendah_10lux", "target": 15},
    "3": {"nama": "Test - Sedang 50 Lux",  "sub_folder": "test/sedang_50lux", "target": 15},
    "4": {"nama": "Test - Tinggi 150 Lux", "sub_folder": "test/tinggi_150lux", "target": 15}
}

def cetak_menu():
    print("\n" + "="*50)
    print("      MENU AKUISISI DATASET CITRA WAJAH")
    print("="*50)
    for kunci, item in KONDISI_MENU.items():
        print(f" [{kunci}] {item['nama']} (Target: {item['target']} Gambar)")
    print(" [0] Keluar Aplikasi")
    print("="*50)

def pilih_subjek():
    print("\n Pilih Subjek (Person):")
    for idx, person in enumerate(NAMA_PERSON, 1):
        print(f" [{idx}] {person}")
    while True:
        pilihan = input("Masukkan nomor person (1-5): ").strip()
        if pilihan in [str(i) for i in range(1, 6)]:
            return NAMA_PERSON[int(pilihan) - 1]
        print("Pilihan tidak valid. Silakan pilih 1 sampai 5.")

def main():
    while True:
        cetak_menu()
        pilihan_mode = input("Pilih Mode Pengambilan Data: ").strip()
        
        if pilihan_mode == "0":
            print("\nTerima kasih. Semangat skripsiannya!")
            break
            
        if pilihan_mode not in KONDISI_MENU:
            print("Pilihan mode tidak valid. Silakan coba lagi.")
            continue
            
        mode_terpilih = KONDISI_MENU[pilihan_mode]
        subjek_terpilih = pilih_subjek()
        
        # Penentuan folder penyimpanan akhir
        target_dir = os.path.join(BASE_PATH, mode_terpilih["sub_folder"], subjek_terpilih)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\n[INFO] Menyiapkan Kamera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Kamera tidak dapat diakses!")
            continue
            
        # ── 1. FASE WARM-UP KAMERA ──────────────────────────────────────────
        print(f"[INFO] Fase Warm-up Sensor Kamera ({WARMUP_FRAMES} frame)...")
        for _ in range(WARMUP_FRAMES):
            ret, frame = cap.read()
            if not ret:
                break
            # Tampilkan visual count down warm-up di layar
            cv2.putText(frame, "Stabilisasi Kamera... Bersiaplah!", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Akuisisi Citra - Warm Up", frame)
            cv2.waitKey(1)
            
        cv2.destroyWindow("Akuisisi Citra - Warm Up")
        
        # ── 2. FASE PENGAMBILAN GAMBAR (CAPTURE) ───────────────────────────
        print(f"[READY] Mulai capture otomatis untuk {subjek_terpilih} sebanyak {mode_terpilih['target']} gambar.")
        print("Tekan [Q] di jendela kamera untuk membatalkan proses.")
        
        count = 0
        waktu_terakhir = time.time()
        
        while count < mode_terpilih["target"]:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Gagal membaca frame dari kamera.")
                break
                
            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Deteksi wajah menggunakan Haar Cascade
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
            )
            
            waktu_sekarang = time.time()
            
            # Gambar bounding box untuk panduan subjek di layar
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 256, 0), 2)
                
                # Cek apakah delay waktu terpenuhi untuk melakukan capture gambar berikutnya
                if waktu_sekarang - waktu_terakhir >= CAPTURE_DELAY:
                    count += 1
                    # Crop wajah, ubah ke grayscale, dan resize ke skala (100, 100)
                    face_roi = gray[y:y+h, x:x+w]
                    face_resized = cv2.resize(face_roi, IMG_SIZE)
                    
                    # Beri nama file unik berdasarkan timestamp agar tidak tumpang tindih
                    timestamp = int(time.time() * 1000)
                    file_name = f"{subjek_terpilih}_{count}_{timestamp}.jpg"
                    file_path = os.path.join(target_dir, file_name)
                    
                    cv2.imwrite(file_path, face_resized)
                    waktu_terakhir = waktu_sekarang
                    print(f"[{count}/{mode_terpilih['target']}] Tersimpan: {file_name}")
                    
                    # Flash effect hijau pada box saat berhasil capture
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), -1)
                    
            # Tampilkan informasi status real-time pada frame GUI
            info_text = f"Mode: {mode_terpilih['nama']}"
            progress_text = f"Progress: {count}/{mode_terpilih['target']} Gambar"
            cv2.putText(display_frame, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, progress_text, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Subjek: {subjek_terpilih}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
            
            cv2.imshow("Proses Pengambilan Dataset", display_frame)
            
            # Berhenti jika menekan tombol 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Pengambilan gambar dihentikan oleh pengguna.")
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[SUKSES] Selesai memproses {subjek_terpilih} di folder: {target_dir}")
        input("\nTekan [Enter] untuk kembali ke Menu Utama...")

if __name__ == "__main__":
    main()