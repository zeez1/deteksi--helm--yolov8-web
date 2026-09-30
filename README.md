# Penerapan YOLOv8 untuk Deteksi Helm Pengendara Sepeda Motor Berbasis Web

## Tujuan penelitian

Penelitian ini menerapkan YOLOv8 untuk mendeteksi penggunaan helm pada pengendara sepeda motor sebagai dasar aplikasi berbasis web. Implementasi website **belum dilakukan**.

## Metode

Model deteksi yang digunakan adalah YOLOv8 dari Ultralytics. Eksperimen baseline M0 menggunakan `yolov8n.pt` tanpa augmentasi, dengan dataset original yang telah dibagi menjadi train, validation, dan test.

## Dataset

Dataset YOLO berisi 908 gambar dan 908 label, terbagi menjadi 635 train, 181 validation, dan 92 test. Dataset memiliki dua kelas:

- `0`: helmet
- `1`: no-helmet

Audit anotasi dan validasi pembagian dataset selesai. Dataset, ZIP sumber, gambar, dan label tidak disimpan di repository ini; lihat `.gitignore`.

## Struktur proyek

```text
.
├── docs/
│   └── progress.md
├── scripts/
│   ├── prepare_dataset.py
│   ├── smoke_test_yolov8.py
│   └── train_m0_baseline.py
├── .gitignore
└── README.md
```

Folder lokal `dataset/`, `experiments/`, `reports/`, `runs/`, dan `weights/` diabaikan Git. Artefak dataset dan model besar tidak dimasukkan ke GitHub.

## Status pengerjaan

- Audit dan persiapan dataset: selesai.
- Smoke test YOLOv8n 2 epoch: berhasil.
- Training baseline M0 YOLOv8n 50 epoch tanpa augmentasi: selesai.
- Evaluasi validation dan test M0: selesai.
- Implementasi website: belum dilakukan.
- Eksperimen M1–M5: belum dilakukan.

Ringkasan progres dan metrik yang benar-benar tercatat ada di [docs/progress.md](docs/progress.md). Repository ini belum memiliki commit awal; file yang akan di-commit dapat diperiksa dengan `git status`.
