# Penerapan YOLOv8 untuk Deteksi Helm Pengendara Sepeda Motor Berbasis Web

## Deskripsi

Penelitian ini menerapkan model deteksi objek YOLOv8 untuk mengenali pengendara sepeda motor yang menggunakan atau tidak menggunakan helm. Model akan menjadi komponen deteksi pada aplikasi berbasis web.

## Tujuan penelitian

- Menyiapkan dan memvalidasi dataset deteksi helm.
- Menerapkan YOLOv8 untuk mendeteksi kelas `helmet` dan `no-helmet`.
- Mengevaluasi model dan menggunakan hasilnya sebagai dasar pengembangan aplikasi web.

## Metode

Penelitian menggunakan YOLOv8 dari Ultralytics. Baseline M0 menggunakan model YOLOv8n (`yolov8n.pt`) tanpa augmentasi, dengan dataset yang dibagi menjadi train, validation, dan test. Catatan pelaksanaan dan metrik eksperimen yang sudah tercatat tersedia di [docs/progress.md](docs/progress.md).

## Dataset

Dataset terdiri atas 908 gambar dan 908 label: 635 data train, 181 validation, dan 92 test. Dataset memiliki dua kelas:

- `0`: `helmet`
- `1`: `no-helmet`

Audit anotasi dan validasi pembagian dataset telah dicatat selesai. Arsip sumber dataset lokal berukuran sekitar 260 MB. Dataset, arsip ZIP, gambar, dan label tidak disertakan di repository; aturan pengabaiannya ada di [.gitignore](.gitignore).

## Progress saat ini

- Audit dan pembagian dataset: selesai.
- Smoke test YOLOv8n: selesai.
- Training baseline M0 dan evaluasi validation/test: selesai menurut catatan eksperimen.
- Eksperimen M1–M5: belum dilakukan.
- Implementasi aplikasi/website: belum dilakukan.
- Analisis perbandingan dan kesimpulan penelitian: belum dilakukan.

Tidak ada klaim hasil eksperimen baru yang ditambahkan dalam README ini. Rincian hasil yang sudah tercatat beserta konteksnya dapat dilihat di [docs/progress.md](docs/progress.md).

## Rencana tahap berikutnya

1. Menentukan dan menjalankan eksperimen lanjutan M1–M5.
2. Membandingkan hasil eksperimen berdasarkan evaluasi validation dan test.
3. Menetapkan model yang akan digunakan pada aplikasi.
4. Mengimplementasikan dan menguji aplikasi deteksi berbasis web.
5. Menyusun analisis, kesimpulan, dan dokumentasi penelitian.

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

Folder lokal `dataset/`, `experiments/`, `reports/`, `runs/`, dan `weights/`, lingkungan virtual, serta berkas model dan arsip ZIP diabaikan Git. Repository sudah memiliki commit awal; perubahan dokumentasi/configuration berikutnya dapat ditinjau melalui `git status`. Tidak ada perubahan yang di-commit atau di-push sebagai bagian dari persiapan ini.
