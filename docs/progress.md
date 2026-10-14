# Logbook dan audit progres penelitian

Judul penelitian: **Analisis Pengaruh Augmentasi Data terhadap Performa YOLOv8 pada Deteksi Penggunaan Helm Pengendara Sepeda Motor Berbasis Web.**

Status di bawah disusun dari file yang ditemukan di workspace pada 2026-10-14. Keberadaan script saja tidak dianggap sebagai bukti training: status training dicocokkan dengan konfigurasi, `results.csv`, bobot, serta laporan evaluasi. Folder `dataset/`, `experiments/`, `reports/`, `runs/`, `weights/`, file model, dan file ZIP diabaikan Git secara umum. Satu pengecualian yang dipilih untuk repository adalah checkpoint `experiments/M5_controlled_augmentation/weights/best.pt`; file lain yang menjadi bukti eksperimen tetap lokal dan tidak disertakan.

Validasi dokumentasi: `python -m unittest discover -s tests -v` lulus untuk 9 unit test (M1/M4); `git diff --check` juga lulus. Folder `runs/detect/` ditemukan kosong saat audit.

## Dataset dan setup

- Dataset split penelitian ada di `dataset/helmet_research/`, dengan `data.yaml` berisi kelas `0 = helmet` dan `1 = no-helmet`.
- Laporan `reports/final_split_report.txt` menyatakan 635 train, 181 validation, dan 92 test; 908 gambar dan label berpasangan; 3.309 objek (2.024 helmet, 1.285 no-helmet); 0 mismatch, 0 class ID tidak valid, dan tidak ada grup duplikat lintas split.
- Dataset sumber lokal `dataset/yolov8-helmet-final-use-no-augmentation.v2-no-augmentasi.yolov8.zip` berukuran 272.924.564 byte. Dataset gambar/label tidak direncanakan untuk dimasukkan ke repository.
- Base checkpoint YOLOv8n `yolov8n.pt` ada di root dan dirujuk oleh training M0–M5 serta smoke test.
- Smoke test 2 epoch ada di `experiments/smoke_test/yolov8n/`; laporan M0–M5 menunjukkan training 50 epoch per tahap, seed 42, imgsz 640, batch 4, dan test set tidak digunakan untuk training.

## Status eksperimen M0–M5

Semua angka evaluasi di tabel berikut berasal dari `reports/M0_M1_M2_M3_M4_M5_comparison.csv` dan laporan test masing-masing (`reports/M*_test_results.json`). Laporan mencatat 92 gambar test per tahap. Nilai ditampilkan sampai enam angka desimal; bukan hasil perhitungan ulang dari grafik.

| Tahap / status | Metode yang ditemukan | Script utama | Dataset / training yang terbukti | Evaluasi test: P / R / mAP50 / mAP50–95 |
|---|---|---|---|---|
| **M0 — selesai menurut artefak** | Baseline YOLOv8n; augmentasi YOLO tambahan dinonaktifkan. | `scripts/train_m0_baseline.py` | `dataset/helmet_research/data.yaml`; konfigurasi 50 epoch, `augment=False`, operasi augmentasi dinonaktifkan; `experiments/M0_baseline/results.csv`, `args.yaml`, `weights/best.pt`, dan `last.pt` tersedia. | 0.817110 / 0.708094 / 0.766207 / 0.377967 |
| **M1 — selesai menurut artefak** | Horizontal flip offline pada data train; konfigurasi mencatat probabilitas flip 1.0 sebelum training, dan augmentasi online dinonaktifkan. | `scripts/create_m1_flip_dataset.py`, `scripts/audit_m1_dataset.py`, `scripts/train_m1_flip.py` | `dataset/experiments/M1_flip/data.yaml`; audit M1, konfigurasi 50 epoch, `results.csv`, dua checkpoint, dan laporan test tersedia. | 0.709798 / 0.678256 / 0.697354 / 0.325731 |
| **M2 — selesai menurut artefak** | Rotasi offline dengan sudut acak dalam rentang −15° hingga +15° pada train; validasi/test tidak diaugmentasi. Augmentasi online dinonaktifkan. | `scripts/create_m2_rotation_dataset.py`, `scripts/train_m2_rotation.py` | `dataset/experiments/M2_rotation/data.yaml`; laporan audit M2, konfigurasi 50 epoch, `results.csv`, checkpoint, dan evaluasi test tersedia. | 0.786432 / 0.696881 / 0.749013 / 0.338467 |
| **M3 — selesai menurut artefak** | Brightness factor 0.8 dan contrast factor 1.2 offline pada train; transformasi bounding box tidak diperlukan; augmentasi online dinonaktifkan. | `scripts/create_m3_brightness_contrast_dataset.py`, `scripts/train_m3_brightness_contrast.py` | `dataset/experiments/M3_brightness_contrast/data.yaml`; audit M3, konfigurasi 50 epoch, `results.csv`, checkpoint, dan evaluasi test tersedia. | 0.843268 / 0.726593 / 0.790732 / 0.387333 |
| **M4 — selesai menurut artefak** | Implementasi aktual adalah kombinasi offline horizontal flip, rotasi −15°…+15°, brightness 0.8, dan contrast 1.2 pada train. Tidak ada augmentasi online tambahan. | `scripts/create_m4_compound_dataset.py`, `scripts/audit_m4_dataset.py`, `scripts/train_m4_compound.py` | `dataset/experiments/M4_compound/data.yaml`; audit M4, konfigurasi bertanda `training_status: completed`, 50 epoch di `results.csv`, checkpoint, dan laporan test tersedia. | 0.755783 / 0.636515 / 0.683067 / 0.308553 |
| **M5 — selesai menurut artefak** | Augmentasi online terkontrol saat training: horizontal flip p=0.5; rotasi ±5°; translate 0.05; scale 0.1; Albumentations `RandomBrightnessContrast` limit ±0.1 dengan p=0.2. Mosaic, mixup, HSV, dan augmentasi lain yang terdaftar dinonaktifkan. | `scripts/train_m5_controlled_augmentation.py` (`--preflight`, `--train`) | Dataset asli `dataset/helmet_research/data.yaml`; konfigurasi mencatat preflight lulus, training selesai 50 epoch, seed 42, `results.csv`, checkpoint, dan evaluasi test tersedia. `app.py` juga menunjuk ke `best.pt` M5. | 0.885168 / 0.761842 / 0.837365 / 0.419074 |

`P` = precision dan `R` = recall. F1 dan waktu inferensi tersedia di beberapa laporan tahap, tetapi tidak dicantumkan pada tabel perbandingan ini karena bentuk laporannya tidak seragam untuk semua tahap. Metrik disajikan sebagai nilai laporan yang tersedia, tanpa klaim uji signifikansi atau generalisasi di luar dataset tersebut.

### Catatan interpretasi dan bukti

- Smoke test 2 epoch bukan M0 dan tidak dihitung sebagai eksperimen pembanding; script-nya `scripts/smoke_test_yolov8.py`.
- Setiap tahap eksperimen melaporkan evaluasi pada split test yang sama (92 gambar). Test tidak digunakan sebagai data training menurut konfigurasi, tetapi penggunaan test berulang untuk membandingkan/memilih antar-M0–M5 dapat membuat estimasi akhir kurang independen. Pertimbangkan validasi terpisah atau test eksternal yang belum pernah dipakai sebelum menyatakan performa final.
- Hasil test terbaik dalam CSV perbandingan adalah M5 pada mAP50 dan mAP50–95. Itu hanya ringkasan artefak lokal, bukan klaim bahwa perbedaannya signifikan.
- M5 memberi `summary_vs_M3` dan `best_models_M0_M5` di laporan lokalnya. Nilai tabel tetap bersumber dari CSV perbandingan dan laporan test, bukan pernyataan otomatis di kode.
- Checkpoint, konfigurasi rinci, `results.csv`, audit JSON/TXT, serta plot hasil tidak ikut dalam Git saat ini. Jika dokumentasi ini dipublikasikan tanpa artefak tersebut, pembaca tidak dapat mereproduksi verifikasi metrik hanya dari repository.

## Script yang tersedia

| Kegunaan | File |
|---|---|
| Membagi/mengaudit dataset dasar | `scripts/prepare_dataset.py` |
| Audit dataset M1 dan M4 | `scripts/audit_m1_dataset.py`, `scripts/audit_m4_dataset.py` |
| Membentuk dataset M1–M4 | `scripts/create_m1_flip_dataset.py`, `scripts/create_m2_rotation_dataset.py`, `scripts/create_m3_brightness_contrast_dataset.py`, `scripts/create_m4_compound_dataset.py` |
| Training M0–M5 | `scripts/train_m0_baseline.py`, `scripts/train_m1_flip.py`, `scripts/train_m2_rotation.py`, `scripts/train_m3_brightness_contrast.py`, `scripts/train_m4_compound.py`, `scripts/train_m5_controlled_augmentation.py` |
| Smoke test | `scripts/smoke_test_yolov8.py` |
| Inferensi M5 pada satu gambar | `scripts/test_m5_inference.py` |
| Grafik perbandingan dan diagram alur | `scripts/create_m0_m5_comparison_chart.py`, `scripts/create_research_flowchart.py` |

Unit test yang ditemukan: `tests/test_m1_flip_validation.py` dan `tests/test_m4_compound_validation.py`. Test M1 memeriksa validasi bounding box dan transformasi flip; test M4 memeriksa transformasi bounding box gabungan/rotasi.

## Laporan dan keluaran yang ditemukan

Semua jalur di bagian ini merupakan artefak lokal dan diabaikan Git; pengecualian yang direncanakan hanya checkpoint `best.pt` M5 yang tercantum pada inventaris di bawah.

- Dataset: `reports/dataset_audit.json`, `reports/dataset_audit.txt`, `reports/final_split_report.json`, `reports/final_split_report.txt`, `reports/class_distribution.csv`.
- Per tahap: `reports/M0_config.json` hingga `reports/M5_config.json`; `reports/M0_test_results.json` hingga `reports/M5_test_results.json`; laporan TXT tersedia untuk test M0–M5. Audit dataset M1–M4 tersedia sebagai JSON/TXT.
- Perbandingan: `reports/M0_M1_M2_comparison.csv`, `reports/M0_M1_M2_M3_comparison.csv`, `reports/M0_M1_M2_M3_M4_comparison.csv`, `reports/M0_M1_M2_M3_M4_M5_comparison.csv`, `reports/M0_vs_M1.csv`.
- Visualisasi: `reports/grafik_perbandingan_m0_m5.png/.pdf` dan `reports/diagram_alur_penelitian.png/.pdf`.
- Log training setiap eksperimen ada di folder `experiments/<tahap>/`, terutama `args.yaml` dan `results.csv`, beserta plot dan bobot.

## Inventaris file model lokal

Ukuran MiB di bawah dihitung sebagai byte / 1.048.576. Pasangan `best.pt`/`last.pt` untuk masing-masing tahap memiliki ukuran sama, tetapi tetap dicatat sebagai dua file. Asal tiap checkpoint ditautkan ke nama folder eksperimen, konfigurasi, dan log yang tersedia.

| File relatif | Ukuran (byte) | Ukuran (MiB) | Asal / klasifikasi | Dipakai web? |
|---|---:|---:|---|---|
| `yolov8n.pt` | 6,549,796 | 6.25 | Bobot dasar YOLOv8n; sumber training M0–M5 dan smoke test. | Tidak |
| `weights/yolo26n.pt` | 5,544,453 | 5.29 | Bobot model lokal lain; tidak dirujuk script training/inferensi yang diperiksa, tahap asal tidak terverifikasi. | Tidak |
| `experiments/smoke_test/yolov8n/weights/best.pt` | 6,220,138 | 5.93 | Checkpoint smoke test 2 epoch, bukan model eksperimen M0. | Tidak |
| `experiments/smoke_test/yolov8n/weights/last.pt` | 6,220,138 | 5.93 | Checkpoint terakhir smoke test 2 epoch. | Tidak |
| `experiments/M0_baseline/weights/best.pt` | 6,226,218 | 5.94 | Bobot terpilih eksperimen M0. | Tidak |
| `experiments/M0_baseline/weights/last.pt` | 6,226,218 | 5.94 | Bobot epoch terakhir M0. | Tidak |
| `experiments/M1_flip/weights/best.pt` | 6,226,474 | 5.94 | Bobot terpilih eksperimen M1. | Tidak |
| `experiments/M1_flip/weights/last.pt` | 6,226,474 | 5.94 | Bobot epoch terakhir M1. | Tidak |
| `experiments/M2_rotation/weights/best.pt` | 6,226,666 | 5.94 | Bobot terpilih eksperimen M2. | Tidak |
| `experiments/M2_rotation/weights/last.pt` | 6,226,666 | 5.94 | Bobot epoch terakhir M2. | Tidak |
| `experiments/M3_brightness_contrast/weights/best.pt` | 6,226,666 | 5.94 | Bobot terpilih eksperimen M3. | Tidak |
| `experiments/M3_brightness_contrast/weights/last.pt` | 6,226,666 | 5.94 | Bobot epoch terakhir M3. | Tidak |
| `experiments/M4_compound/weights/best.pt` | 6,226,730 | 5.94 | Bobot terpilih eksperimen M4. | Tidak |
| `experiments/M4_compound/weights/last.pt` | 6,226,730 | 5.94 | Bobot epoch terakhir M4. | Tidak |
| `experiments/M5_controlled_augmentation/weights/best.pt` | 6,232,106 | 5.94 | Bobot terpilih M5; menjadi checkpoint yang dirujuk `app.py`. | **Ya** |
| `experiments/M5_controlled_augmentation/weights/last.pt` | 6,232,106 | 5.94 | Bobot epoch terakhir M5. | Tidak; aplikasi menunjuk `best.pt`. |

Semua file model yang ditemukan jauh di bawah batas 50 MB yang menjadi ambang peringatan GitHub untuk file biasa, dan di bawah batas keras 100 MB per file. Untuk kebutuhan aplikasi yang ada, checkpoint `experiments/M5_controlled_augmentation/weights/best.pt` dipilih sebagai satu-satunya model untuk disertakan melalui Git biasa. Pola `.gitignore` untuk `experiments/` dan `*.pt` tetap berlaku pada file model lainnya. Git LFS tidak diperlukan untuk satu checkpoint sekitar 5.94 MiB; Git LFS terpasang di mesin tetapi belum dikonfigurasi repo (`.gitattributes` tidak ditemukan dan atribut `filter` untuk `*.pt` tidak ditetapkan). GitHub Release bisa dipilih bila nanti ingin membagikan banyak checkpoint tanpa menambahkannya ke riwayat repository.

Bobot `best.pt` M5 diperlukan aplikasi lokal saat ini. Bobot pretrained `yolov8n.pt` diperlukan untuk menjalankan ulang training; `yolo26n.pt` tidak diketahui memiliki fungsi dalam pipeline yang diperiksa. Dataset dan arsip 272.924.564 byte, cache, `.venv/`, log/hasil training, gambar contoh lokal, file rahasia, dan checkpoint lain tidak disarankan masuk repository publik tanpa kebutuhan/izin tersendiri.

## Petunjuk menjalankan

Jalankan dari root repository dengan environment Python yang menyediakan dependensi terkait. `requirements-m5.txt` hanya mencantumkan `albumentations==1.4.24`; `requirements-web.txt` hanya mencantumkan Streamlit, sehingga kedua file itu belum merupakan lockfile environment lengkap.

```powershell
python -m unittest discover -s tests -v
python scripts/prepare_dataset.py
python scripts/smoke_test_yolov8.py
python scripts/train_m0_baseline.py
python scripts/create_m1_flip_dataset.py
python scripts/audit_m1_dataset.py
python scripts/train_m1_flip.py
python scripts/create_m2_rotation_dataset.py
python scripts/train_m2_rotation.py
python scripts/create_m3_brightness_contrast_dataset.py
python scripts/train_m3_brightness_contrast.py
python scripts/create_m4_compound_dataset.py
python scripts/audit_m4_dataset.py
python scripts/train_m4_compound.py
python scripts/train_m5_controlled_augmentation.py --preflight
python scripts/train_m5_controlled_augmentation.py --train
python scripts/create_m0_m5_comparison_chart.py
python scripts/create_research_flowchart.py
```

Persiapan dataset memerlukan data sumber lokal dan training memerlukan bobot dasar. Periksa output/guard script sebelum mengulang pembuatan dataset atau training. M5 menerapkan preflight sebelum mode `--train`.

Aplikasi web menggunakan Streamlit, PyTorch, Ultralytics, OpenCV, Pillow, checkpoint M5 lokal, dan FFmpeg untuk video. Pastikan dependensi runtime lengkap serta checkpoint ada, lalu jalankan:

```powershell
python -m pip install -r requirements-web.txt
python -m streamlit run app.py
```

`requirements-web.txt` hanya memasang Streamlit; perintah di atas tidak memasang dependensi lain yang dibutuhkan app. Foto JPG/JPEG/PNG dan video MP4/AVI/MOV didukung. Aplikasi memilih CUDA device 0 jika ada dan CPU sebagai fallback; file upload/video sementara tidak disimpan ke folder dataset atau eksperimen. Untuk inferensi gambar satu file, `scripts/test_m5_inference.py --source path\ke\gambar.jpg --conf 0.25` menyimpan hasil ke `inference/M5/run_<id>/`.

## Pekerjaan yang masih perlu dilakukan

1. Uji dan dokumentasikan setup dari clone bersih, termasuk daftar dependensi training/inferensi yang lengkap dan cara memperoleh dataset/model lokal.
2. Periksa lisensi/izin redistribusi dataset/model sebelum membagikan artefak penelitian lebih lanjut.
3. Jika mengunggah laporan eksperimen, pilih konfigurasi, audit, dan hasil yang mendukung reproduksibilitas tanpa dataset besar atau informasi path absolut mesin; `.gitignore` saat ini mengabaikan seluruh `reports/` dan `experiments/`.
4. Tambahkan bukti pengujian aplikasi Streamlit/web jika aplikasi akan diklaim sudah tervalidasi; file test saat ini hanya unit test transformasi label serta script inferensi gambar manual.
5. Tinjau efek penggunaan split test yang sama untuk enam tahap dan hindari menggunakannya berulang untuk pemilihan model final; pertimbangkan test eksternal/untouched untuk estimasi akhir.
6. Lengkapi analisis penelitian, pembahasan keterbatasan, dan kesimpulan berdasarkan protokol evaluasi yang ditetapkan.
