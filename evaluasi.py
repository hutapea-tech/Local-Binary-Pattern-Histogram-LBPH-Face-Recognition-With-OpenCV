"""
Kode evaluasi.py — Evaluasi LBPH per Kondisi Pencahayaan (Output Lengkap)
=====================================================================
Judul  : Analisis Pengaruh Intensitas Pencahayaan Terhadap Akurasi LBPH
Penulis: Adi Saputra Hutavea

Output: 
- hasil_evaluasi_lengkap.txt (semua tabel, confusion matrix, confidence per gambar)
- hasil_evaluasi.json (data mentah)
- Tiga grafik: akurasi, presisi+recall, rata-rata confidence
"""

import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

# ── KONFIGURASI ──────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
OUTPUT_DIR   = "output_evaluasi"
THRESHOLD    = 70.0
IMG_SIZE     = (100, 100)
NAMA_PERSON  = [f"person_{i}" for i in range(1, 6)]
KONDISI_TEST = {
    "Rendah 10 lux"   : {"folder": "rendah_10lux", "lux": 10},
    "Sedang 50 lux"  : {"folder": "sedang_50lux", "lux": 50},
    "Tinggi 150 lux"  : {"folder": "tinggi_150lux", "lux": 150},
}
LBPH_PARAMS  = dict(radius=1, neighbors=8, grid_x=8, grid_y=8)

# Pemetaan untuk tampilan
KONDISI_NAMA = {v["folder"]: k for k, v in KONDISI_TEST.items()}

# ── UTILITAS ─────────────────────────────────────────────────────────────────
def muat_gambar_train(folder):
    """Muat data training: hanya gambar dan label (tanpa nama file)"""
    images, labels = [], []
    for lbl, person in enumerate(NAMA_PERSON):
        path = os.path.join(folder, person)
        if not os.path.isdir(path):
            continue
        for f in sorted(os.listdir(path)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(cv2.resize(img, IMG_SIZE))
            labels.append(lbl)
    return images, labels

def muat_gambar_test(folder):
    """Muat data testing: gambar, label, dan nama file"""
    data = []  # list of (img, label, filename)
    for lbl, person in enumerate(NAMA_PERSON):
        path = os.path.join(folder, person)
        if not os.path.isdir(path):
            continue
        for f in sorted(os.listdir(path)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            data.append((cv2.resize(img, IMG_SIZE), lbl, f))
    return data

def hitung_metrik_dengan_unknown(y_true, y_pred, distances, n_kelas):
    """
    Hitung metrik termasuk kelas 'unknown' (indeks n_kelas).
    y_pred: -1 untuk unknown, selain itu 0..n_kelas-1.
    """
    # Confusion matrix size (n_kelas+1) x (n_kelas+1)
    # Baris/kolom terakhir = unknown
    cm = np.zeros((n_kelas + 1, n_kelas + 1), dtype=int)
    for t, p in zip(y_true, y_pred):
        if p == -1:
            p = n_kelas
        cm[t][p] += 1

    # Hitung akurasi (hanya gambar yang dikenali? Tidak, semua termasuk unknown)
    n_total = len(y_true)
    n_benar = sum(1 for t, p in zip(y_true, y_pred) if p != -1 and p == t)
    akurasi = n_benar / n_total if n_total else 0.0

    # Precision, recall per kelas (0..n_kelas)
    prec_list, rec_list = [], []
    per_kelas = {}
    for k in range(n_kelas + 1):
        tp = cm[k][k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        prec_list.append(prec)
        rec_list.append(rec)
        if k < n_kelas:
            per_kelas[NAMA_PERSON[k]] = {"presisi": prec, "recall": rec}
        else:
            per_kelas["unknown"] = {"presisi": prec, "recall": rec}

    return {
        "akurasi"       : akurasi,
        "presisi_macro" : float(np.mean(prec_list[:n_kelas])),  # tanpa unknown
        "recall_macro"  : float(np.mean(rec_list[:n_kelas])),
        "avg_confidence": float(np.mean(distances)) if distances else 0.0,
        "n_total"       : n_total,
        "n_benar"       : n_benar,
        "n_salah"       : n_total - n_benar - (y_pred.count(-1)),
        "n_ditolak"     : y_pred.count(-1),
        "pct_ditolak"   : y_pred.count(-1) / n_total * 100 if n_total else 0.0,
        "per_kelas"     : per_kelas,
        "cm"            : cm.tolist(),
        "prec_all"      : prec_list,   # termasuk unknown
        "rec_all"       : rec_list,
    }

def hitung_tp_tn_fp_fn_per_kelas(y_true, y_pred, n_kelas):
    """Hitung TP, TN, FP, FN untuk setiap kelas (0..n_kelas-1)"""
    n_total = len(y_true)
    hasil = []
    for k in range(n_kelas):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == k and p == k)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == k and p != k)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != k and p == k)
        tn = n_total - tp - fp - fn
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        acc  = (tp + tn) / n_total if n_total else 0.0
        hasil.append({
            "kelas": NAMA_PERSON[k],
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "precision": prec, "recall": rec, "accuracy": acc
        })
    return hasil

# ── EVALUASI UTAMA ───────────────────────────────────────────────────────────
def evaluasi():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Muat & latih model
    train_imgs, train_labels = muat_gambar_train(os.path.join(DATASET_DIR, "train"))
    kelas_aktif = sorted(set(train_labels))
    n_kelas = len(NAMA_PERSON)
    print(f"\nData train: {len(train_imgs)} gambar, {n_kelas} orang")

    model = cv2.face.LBPHFaceRecognizer_create(**LBPH_PARAMS)
    model.train(train_imgs, np.array(train_labels))
    print(f"Model dilatih. Threshold: {THRESHOLD}\n")

    # Menyimpan semua hasil per kondisi
    hasil_kondisi = {}
    all_cm = np.zeros((n_kelas + 1, n_kelas + 1), dtype=int)  # akumulasi cm keseluruhan
    all_details = []  # untuk keperluan precision/recall keseluruhan (y_true, y_pred)

    for kondisi_key, info in KONDISI_TEST.items():
        test_folder = os.path.join(DATASET_DIR, "test", info["folder"])
        test_data = muat_gambar_test(test_folder)
        if not test_data:
            print(f"[SKIP] {kondisi_key} — tidak ada data di {test_folder}")
            continue

        y_true = []
        y_pred = []
        distances = []
        detail_gambar = []  # (idx, nama_subjek, filename, pred_label, confidence, status)

        print(f"\n>>> Memproses kondisi: {kondisi_key.upper()} ({info['lux']} lux) ...")

        for idx, (img, true_lbl, fname) in enumerate(test_data):
            pred_raw, dist = model.predict(img)
            pred_final = pred_raw if dist <= THRESHOLD else -1
            y_true.append(true_lbl)
            y_pred.append(pred_final)
            distances.append(dist)

            nama_true = NAMA_PERSON[true_lbl]
            nama_pred = NAMA_PERSON[pred_final] if pred_final >= 0 else "unknown"
            status = "BENAR" if (pred_final == true_lbl) else "SALAH"
            detail_gambar.append((idx+1, nama_true, fname, nama_pred, dist, status))

        # Hitung metrik dengan unknown
        metrik = hitung_metrik_dengan_unknown(y_true, y_pred, distances, n_kelas)
        metrik["kondisi"] = kondisi_key
        metrik["lux"] = info["lux"]
        metrik["detail_gambar"] = detail_gambar
        hasil_kondisi[kondisi_key] = metrik

        # Akumulasi confusion matrix keseluruhan
        cm_this = np.array(metrik["cm"])
        all_cm += cm_this

        # Simpan semua prediksi untuk perhitungan precision/recall keseluruhan
        for t, p in zip(y_true, y_pred):
            all_details.append((t, p))

        # Hitung TP/TN/FP/FN per kelas (tanpa unknown)
        tp_tn_fp_fn = hitung_tp_tn_fp_fn_per_kelas(y_true, y_pred, n_kelas)
        metrik["tp_tn_fp_fn"] = tp_tn_fp_fn

    # ── Hitung metrik keseluruhan dari all_details ──
    y_true_all, y_pred_all = zip(*all_details) if all_details else ([], [])
    metrik_global = hitung_metrik_dengan_unknown(list(y_true_all), list(y_pred_all), [], n_kelas)
    # Hitung TP/TN/FP/FN keseluruhan
    tp_tn_fp_fn_global = hitung_tp_tn_fp_fn_per_kelas(list(y_true_all), list(y_pred_all), n_kelas)

    return hasil_kondisi, metrik_global, tp_tn_fp_fn_global, all_cm

# ── CETAK OUTPUT TEKS LENGKAP ─────────────────────────────────────────────────
def cetak_hasil_lengkap(hasil_kondisi, metrik_global, tp_tn_fp_fn_global, all_cm):
    lines = []
    n_kelas = len(NAMA_PERSON)
    labels_kelas = NAMA_PERSON + ["unknown"]

    # 1. CONFUSION MATRIX (KESELURUHAN)
    lines.append("CONFUSION MATRIX (KESELURUHAN)")
    lines.append("=" * 50)
    # Header
    header = " " * 12 + "".join(f"{label:>10}" for label in labels_kelas)
    lines.append(header)
    for i, label in enumerate(labels_kelas):
        row = f"{label:<12}" + "".join(f"{all_cm[i][j]:>10}" for j in range(len(labels_kelas)))
        lines.append(row)
    lines.append("\n")

    # 2. PRECISION | RECALL (KESELURUHAN)
    lines.append("PRECISION | RECALL (KESELURUHAN)")
    lines.append("=" * 50)
    for i, label in enumerate(labels_kelas):
        p = metrik_global["prec_all"][i]
        r = metrik_global["rec_all"][i]
        lines.append(f"{label:<12}| {p:.2f} | {r:.2f}")
    lines.append("\n")

    # 3. ANALISIS PER PENCAHAYAAN
    lines.append("ANALISIS PER PENCAHAYAAN")
    lines.append("=" * 50)
    lines.append(f"{'Kondisi':<12} | {'Akurasi':>8} | {'Benar/Total':^12} | {'Avg Confidence':>14}")
    lines.append("-" * 50)
    for kondisi, h in hasil_kondisi.items():
        benar = h["n_benar"]
        total = h["n_total"]
        akurasi = h["akurasi"] * 100
        avg_conf = h["avg_confidence"]
        lines.append(f"{kondisi:<12} | {akurasi:>7.2f}% | {benar:>3}/{total:<3}     | {avg_conf:>14.4f}")
    lines.append("\n")

    # 4. Untuk setiap kondisi: confusion matrix, precision/recall, TP/TN/FP/FN, confidence per gambar
    for kondisi, h in hasil_kondisi.items():
        cm = np.array(h["cm"])
        lux = h["lux"]
        lines.append(f"CONFUSION MATRIX — {kondisi.upper()}")
        lines.append("=" * 50)
        header = " " * 12 + "".join(f"{label:>10}" for label in labels_kelas)
        lines.append(header)
        for i, label in enumerate(labels_kelas):
            row = f"{label:<12}" + "".join(f"{cm[i][j]:>10}" for j in range(len(labels_kelas)))
            lines.append(row)
        lines.append("")

        lines.append(f"PRECISION | RECALL — {kondisi.upper()}")
        lines.append("=" * 50)
        for i, label in enumerate(labels_kelas):
            p = h["prec_all"][i]
            r = h["rec_all"][i]
            lines.append(f"{label:<12}| {p:.2f} | {r:.2f}")
        lines.append("")

        # TP | TN | FP | FN per kelas
        lines.append("=" * 70)
        lines.append(f"TP | TN | FP | FN PER KELAS  —  Kondisi: {kondisi.upper()}")
        lines.append("=" * 70)
        lines.append(f"{'Kelas':<12} {'TP':>4} {'TN':>5} {'FP':>5} {'FN':>5} {'Precision':>10} {'Recall':>9} {'Accuracy':>9}")
        lines.append("-" * 70)
        for info in h["tp_tn_fp_fn"]:
            lines.append(f"{info['kelas']:<12} {info['TP']:>4} {info['TN']:>5} {info['FP']:>5} {info['FN']:>5} "
                         f"{info['precision']:>9.2f} {info['recall']:>9.2f} {info['accuracy']*100:>8.2f}%")
        lines.append("")

        # CONFIDENCE PER GAMBAR UJI
        lines.append("=" * 75)
        lines.append(f"CONFIDENCE PER GAMBAR UJI  —  Kondisi: {kondisi.upper()}")
        lines.append("=" * 75)
        lines.append(f"{' No':>4} {'Subjek':<12} {'File':<12} {'Prediksi':<12} {'Confidence':>10} {'Status':<8}")
        lines.append("-" * 75)
        conf_vals = []
        for idx, subjek, fname, pred, conf, status in h["detail_gambar"]:
            lines.append(f"{idx:>4}  {subjek:<12} {fname:<12} {pred:<12} {conf:>10.4f} {status:<8}")
            conf_vals.append(conf)
        avg_c = np.mean(conf_vals)
        min_c = np.min(conf_vals)
        max_c = np.max(conf_vals)
        lines.append(f"\n Avg Confidence ({kondisi}): {avg_c:.4f}")
        lines.append(f" Min Confidence ({kondisi}): {min_c:.4f}")
        lines.append(f" Max Confidence ({kondisi}): {max_c:.4f}")
        lines.append("\n" + "="*75 + "\n")

    # Simpan ke file
    output_txt = os.path.join(OUTPUT_DIR, "hasil_evaluasi_lengkap.txt")
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nOutput teks lengkap disimpan: {output_txt}")

# ── GRAFIK (TERPISAH DENGAN FONT BESAR) ──────────────────────────────────────
def buat_grafik_terpisah(hasil_kondisi):
    kondisi_list = list(hasil_kondisi.keys())
    lux_list = [hasil_kondisi[k]["lux"] for k in kondisi_list]
    akurasi = [hasil_kondisi[k]["akurasi"] * 100 for k in kondisi_list]
    presisi = [hasil_kondisi[k]["presisi_macro"] * 100 for k in kondisi_list]
    recall  = [hasil_kondisi[k]["recall_macro"] * 100 for k in kondisi_list]
    avg_conf = [hasil_kondisi[k]["avg_confidence"] for k in kondisi_list]

    # Set font size lebih besar
    plt.rcParams.update({'font.size': 12})

    # Grafik 1: Akurasi saja
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    bars = ax1.bar(kondisi_list, akurasi, color='#2196F3', width=0.5)
    for bar, val in zip(bars, akurasi):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Akurasi (%)", fontsize=13)
    ax1.set_xlabel("Kondisi Pencahayaan", fontsize=13)
    ax1.set_title("Grafik Akurasi Metode LBPH pada Setiap Kondisi Intensitas Pencahayaan",
                  fontsize=14, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "grafik_akurasi.png"), dpi=150)
    plt.close()

    # Grafik 2: Presisi dan Recall (grouped bar)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(kondisi_list))
    width = 0.35
    bars1 = ax2.bar(x - width/2, presisi, width, label='Presisi', color='#4CAF50')
    bars2 = ax2.bar(x + width/2, recall, width, label='Recall', color='#FF9800')
    for bar in bars1:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%", ha='center', fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%", ha='center', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(kondisi_list)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Nilai (%)", fontsize=13)
    ax2.set_xlabel("Kondisi Pencahayaan", fontsize=13)
    ax2.set_title("Grafik Presisi dan Recall (Sensitivitas) Metode LBPH per Kondisi Pencahayaan",
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "grafik_presisi_recall.png"), dpi=150)
    plt.close()

    # Grafik 3: Rata-rata Confidence (lebih rendah = lebih yakin)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    bars = ax3.bar(kondisi_list, avg_conf, color=['#EF5350', '#FFA726', '#66BB6A'], width=0.5)
    for bar, val in zip(bars, avg_conf):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax3.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold = {THRESHOLD}')
    ax3.set_ylabel("Rata-rata Jarak Euclidean", fontsize=13)
    ax3.set_xlabel("Kondisi Pencahayaan", fontsize=13)
    ax3.set_title("Grafik Rata-rata Nilai Confidence LBPH per Kondisi Pencahayaan\n(Lebih Rendah = Lebih Yakin)",
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "grafik_avg_confidence.png"), dpi=150)
    plt.close()

    print("Grafik (3 buah) disimpan di folder output_evaluasi/")

# ── SIMPAN HASIL JSON ────────────────────────────────────────────────────────
def simpan_json(hasil_kondisi):
    # Konversi detail_gambar ke bentuk serializable (list of list)
    for k, v in hasil_kondisi.items():
        if "detail_gambar" in v:
            v["detail_gambar"] = [[idx, subjek, fname, pred, conf, status] 
                                   for idx, subjek, fname, pred, conf, status in v["detail_gambar"]]
        if "cm" in v:
            v["cm"] = np.array(v["cm"]).tolist()
    json_path = os.path.join(OUTPUT_DIR, "hasil_evaluasi.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(hasil_kondisi, f, ensure_ascii=False, indent=2)
    print(f"Data JSON disimpan : {json_path}")

# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  EVALUASI LBPH — PENGARUH INTENSITAS PENCAHAYAAN (OUTPUT LENGKAP)")
    print("=" * 60)

    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Folder '{DATASET_DIR}' tidak ditemukan.")
        raise SystemExit(1)

    hasil_kondisi, metrik_global, tp_tn_fp_fn_global, all_cm = evaluasi()

    if not hasil_kondisi:
        print("[ERROR] Tidak ada hasil. Periksa folder dataset/test/.")
        raise SystemExit(1)

    cetak_hasil_lengkap(hasil_kondisi, metrik_global, tp_tn_fp_fn_global, all_cm)
    simpan_json(hasil_kondisi)
    buat_grafik_terpisah(hasil_kondisi)

    print(f"\nSemua output tersimpan di: {OUTPUT_DIR}/")