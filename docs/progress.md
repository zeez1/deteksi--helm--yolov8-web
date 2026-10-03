# Logbook Perkembangan Penelitian

Judul: **Penerapan YOLOv8 untuk Deteksi Helm Pengendara Sepeda Motor Berbasis Web**

Catatan berikut merangkum artefak yang ditemukan di proyek per 2026-09-30. Folder data, model, dan hasil eksperimen bersifat lokal dan diabaikan oleh Git.

## 2026-09-29 — Audit dan persiapan dataset

- Dataset sumber diekstrak dan diaudit: 908 gambar, 908 label.
- Split dataset penelitian: 635 train, 181 validation, 92 test.
- Kelas yang tercatat dalam konfigurasi dataset: `0 = helmet`, `1 = no-helmet`.
- Laporan audit menunjukkan 0 error anotasi, 0 gambar corrupt, 0 mismatch image-label, dan 0 grup gambar duplikat SHA-256.
- Dataset hasil split lolos validasi; augmentasi belum diterapkan.

## 2026-09-29 — Smoke test

- Smoke test YOLOv8n menggunakan `yolov8n.pt` selesai selama 2 epoch di NVIDIA GeForce RTX 2050.
- Training berhasil dan checkpoint serta `results.csv` tersedia.
- Smoke test hanya memverifikasi pipeline; hasilnya bukan hasil eksperimen M0.

## 2026-09-29 — Eksperimen baseline M0

- YOLOv8n (`yolov8n.pt`) dilatih selama 50 epoch pada train split dengan augmentasi dinonaktifkan.
- Evaluasi dilakukan pada validation dan test setelah training. Test set tidak digunakan untuk training.
- Training berlangsung 1.587,45 detik menggunakan NVIDIA GeForce RTX 2050; tercatat tidak ada CUDA out-of-memory.
- Metrik evaluasi test yang tersimpan:

| Metrik | Hasil |
|---|---:|
| Precision | 0,817110 |
| Recall | 0,708094 |
| mAP@50 | 0,766207 |
| mAP@50–95 | 0,377967 |
| F1-score | 0,758706 |
| Inference time | 8,364 ms/gambar |

Checkpoint dan laporan rinci berada di folder eksperimen/laporan lokal dan tidak disertakan dalam repository.

## Status saat ini

- Audit dan split dataset: **selesai**.
- Smoke test: **selesai**.
- Baseline M0 dan evaluasi validation/test: **selesai**.
- Eksperimen M1–M5: **belum dilakukan**.
- Aplikasi/website: **belum dilakukan**.
- Analisis perbandingan dan kesimpulan penelitian: **belum dilakukan**.

## Rencana tahap berikutnya

1. Menentukan dan menjalankan eksperimen lanjutan M1–M5.
2. Membandingkan hasil eksperimen dengan evaluasi validation dan test.
3. Memilih model untuk integrasi aplikasi berbasis web.
4. Mengimplementasikan dan menguji aplikasi deteksi.
5. Menyusun analisis, kesimpulan, dan dokumentasi penelitian.
