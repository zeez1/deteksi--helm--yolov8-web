"""Streamlit app for trying the final M5 helmet-detection model."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import cv2
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
CHECKPOINT = (
    ROOT
    / "experiments"
    / "M5_controlled_augmentation"
    / "weights"
    / "best.pt"
)
IMAGE_SIZES = (640, 960, 1280)
CONFIDENCE_OPTIONS = (0.15, 0.20, 0.25, 0.30, 0.35)
VIDEO_EXTENSIONS = ("mp4", "avi", "mov")


@st.cache_resource(show_spinner="Memuat model M5...")
def load_model(checkpoint_path: str) -> YOLO:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint model M5 tidak ditemukan: {checkpoint}")
    return YOLO(str(checkpoint))


def get_class_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        class_names = {int(class_id): str(name) for class_id, name in names.items()}
    elif isinstance(names, (list, tuple)):
        class_names = {
            class_id: str(name) for class_id, name in enumerate(names)
        }
    else:
        raise TypeError(f"Format model.names tidak didukung: {type(names).__name__}")
    if not class_names:
        raise ValueError("Model M5 tidak memiliki nama kelas.")
    return class_names


def process_video(
    model: YOLO,
    video_data: bytes,
    file_suffix: str,
    confidence: float,
    image_size: int,
    device: int | str,
    progress_bar: Any,
) -> tuple[bytes, dict[str, int | float | bool]]:
    with tempfile.TemporaryDirectory(prefix="yolov8_m5_") as temporary_directory:
        temporary_path = Path(temporary_directory)
        input_path = temporary_path / f"input{file_suffix}"
        intermediate_path = temporary_path / "annotated_intermediate.mp4"
        output_path = temporary_path / "detected.mp4"
        input_path.write_bytes(video_data)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(
                "Video gagal dibaca. Pastikan file valid dan codec video "
                "didukung oleh OpenCV."
            )

        writer = None
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                raise ValueError("Resolusi video tidak dapat dibaca oleh OpenCV.")

            fps_fallback_used = source_fps <= 0
            output_fps = source_fps if source_fps > 0 else 30.0
            writer = cv2.VideoWriter(
                str(intermediate_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                (width, height),
            )
            if writer is None:
                raise RuntimeError(
                    "OpenCV tidak dapat membuat VideoWriter untuk video "
                    "intermediate."
                )
            if not writer.isOpened():
                writer.release()
                writer = None
                raise RuntimeError(
                    "OpenCV tidak dapat membuka VideoWriter untuk video "
                    "intermediate mp4v."
                )

            helmet_count = 0
            no_helmet_count = 0
            processed_frames = 0
            progress_interval = max(1, source_frame_count // 100)

            while True:
                has_frame, frame = capture.read()
                if not has_frame:
                    break

                results = model.predict(
                    source=frame,
                    conf=confidence,
                    imgsz=image_size,
                    device=device,
                    save=False,
                    verbose=False,
                )
                if not results:
                    raise RuntimeError(
                        f"Inference tidak mengembalikan hasil pada frame "
                        f"{processed_frames + 1}."
                    )

                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        if class_id == 0:
                            helmet_count += 1
                        elif class_id == 1:
                            no_helmet_count += 1
                        else:
                            raise ValueError(
                                f"Model menghasilkan class ID {class_id}; "
                                "kelas yang didukung adalah 0 (helmet) dan "
                                "1 (no-helmet)."
                            )

                writer.write(result.plot())
                processed_frames += 1
                if (
                    source_frame_count > 0
                    and processed_frames % progress_interval == 0
                ):
                    progress_bar.progress(
                        min(processed_frames / source_frame_count, 1.0),
                        text=f"Memproses frame {processed_frames} "
                        f"dari {source_frame_count}...",
                    )

            if processed_frames == 0:
                raise ValueError("Video tidak berisi frame yang dapat diproses.")
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if not intermediate_path.is_file() or intermediate_path.stat().st_size == 0:
            raise RuntimeError(
                "OpenCV tidak menghasilkan video intermediate. Codec mp4v "
                "mungkin tidak tersedia atau gagal saat menulis frame."
            )

        progress_bar.progress(0.99, text="Mengonversi video ke MP4 H.264...")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(intermediate_path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-fps_mode",
                    "passthrough",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "FFmpeg tidak ditemukan oleh aplikasi. Pastikan ffmpeg tersedia "
                "di PATH yang digunakan untuk menjalankan Streamlit."
            ) from error
        except subprocess.CalledProcessError as error:
            ffmpeg_error = (error.stderr or "").strip()
            if len(ffmpeg_error) > 2000:
                ffmpeg_error = ffmpeg_error[-2000:]
            raise RuntimeError(
                "FFmpeg gagal mengonversi video ke H.264"
                f"{': ' + ffmpeg_error if ffmpeg_error else '.'}"
            ) from error

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("FFmpeg tidak menghasilkan file video MP4 final.")

        progress_bar.progress(1.0, text="Pemrosesan video selesai.")
        duration = (
            source_frame_count / source_fps
            if source_frame_count > 0 and source_fps > 0
            else 0.0
        )
        statistics: dict[str, int | float | bool] = {
            "frames": processed_frames,
            "helmet": helmet_count,
            "no_helmet": no_helmet_count,
            "width": width,
            "height": height,
            "fps": output_fps,
            "duration": duration,
            "fps_fallback_used": fps_fallback_used,
        }
        return output_path.read_bytes(), statistics


def main() -> None:
    st.set_page_config(page_title="Deteksi Penggunaan Helm", page_icon="🪖", layout="wide")
    st.title("🪖 Deteksi Penggunaan Helm — YOLOv8 M5")

    if not CHECKPOINT.is_file():
        st.error(f"Checkpoint model M5 tidak ditemukan: {CHECKPOINT}")
        st.stop()
    try:
        model = load_model(str(CHECKPOINT))
        names = get_class_names(model.names)
    except Exception as error:
        st.error(f"Gagal memuat model M5: {error}")
        st.stop()

    names_by_normalized = {
        name.strip().casefold(): (class_id, name)
        for class_id, name in names.items()
    }
    if (
        names_by_normalized.get("helmet", (None, ""))[0] != 0
        or names_by_normalized.get("no-helmet", (None, ""))[0] != 1
    ):
        st.error(
            "Urutan kelas model tidak sesuai. Diperlukan 0=helmet dan "
            f"1=no-helmet; model.names={names}"
        )
        st.stop()

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    st.sidebar.header("Pengaturan Inference")
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=CONFIDENCE_OPTIONS[0],
        max_value=CONFIDENCE_OPTIONS[-1],
        value=0.25,
        step=0.05,
    )
    image_size = st.sidebar.selectbox(
        "Image size",
        options=IMAGE_SIZES,
        index=0,
    )
    st.sidebar.caption(
        "Image Size lebih tinggi dapat membantu mendeteksi objek kecil tetapi "
        "membutuhkan waktu inference lebih lama."
    )
    st.sidebar.caption(f"Device: {'CUDA:0' if device == 0 else 'CPU'}")

    with st.expander("Informasi model", expanded=True):
        st.write("Model: YOLOv8 M5")
        st.write("Checkpoint: best.pt")
        st.write("mAP50 TEST: 0.837365")
        st.write("mAP50-95 TEST: 0.419074")
        st.write(f"Class names (model.names): {names}")

    st.info(
        "Upload diproses untuk inference saja, bukan bagian dari perhitungan "
        "metrik TEST. File sementara tidak disimpan ke dataset atau eksperimen."
    )
    input_type = st.radio("Jenis Input", ["Foto", "Video"], horizontal=True)
    if input_type == "Foto":
        uploaded_file = st.file_uploader(
            "Upload foto",
            type=["jpg", "jpeg", "png"],
            help="Format yang didukung: JPG, JPEG, PNG.",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload video",
            type=list(VIDEO_EXTENSIONS),
            help="Format yang didukung: MP4, AVI, MOV.",
        )

    if uploaded_file is None:
        return

    uploaded_data = uploaded_file.getvalue()
    if not uploaded_data:
        st.error("File yang diunggah kosong.")
        return

    if input_type == "Foto":
        try:
            image = Image.open(BytesIO(uploaded_data)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f"Foto tidak dapat dibaca atau formatnya tidak didukung: {error}")
            return

        try:
            with st.spinner("Menjalankan deteksi foto..."):
                results = model.predict(
                    source=image,
                    conf=confidence,
                    imgsz=image_size,
                    device=device,
                    save=False,
                    verbose=False,
                )
        except Exception as error:
            st.error(f"Inference foto gagal: {error}")
            return
        if not results:
            st.error("Model tidak mengembalikan hasil deteksi untuk foto ini.")
            return

        result = results[0]
        counts = {class_id: 0 for class_id in names}
        detections: list[dict[str, Any]] = []
        if result.boxes is not None:
            for index, box in enumerate(result.boxes, start=1):
                class_id = int(box.cls.item())
                if class_id not in names:
                    st.error(
                        f"Model menghasilkan class ID {class_id} yang tidak "
                        f"ada di model.names={names}"
                    )
                    return
                counts[class_id] += 1
                coordinates = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "No": index,
                        "Class": names[class_id],
                        "Confidence": float(box.conf.item()),
                        "Bounding Box": (
                            f"x1={coordinates[0]:.1f}, y1={coordinates[1]:.1f}, "
                            f"x2={coordinates[2]:.1f}, y2={coordinates[3]:.1f}"
                        ),
                    }
                )

        helmet_id, helmet_name = names_by_normalized["helmet"]
        no_helmet_id, no_helmet_name = names_by_normalized["no-helmet"]
        summary_columns = st.columns(3)
        summary_columns[0].metric(helmet_name, counts[helmet_id])
        summary_columns[1].metric(no_helmet_name, counts[no_helmet_id])
        summary_columns[2].metric("Total objek", len(detections))

        original_column, result_column = st.columns(2)
        with original_column:
            st.markdown("#### Foto asli")
            st.image(image, use_container_width=True)
        with result_column:
            st.markdown("#### Hasil deteksi")
            annotated_image = result.plot()[:, :, ::-1]
            st.image(annotated_image, use_container_width=True)

        st.markdown("#### Detail deteksi")
        if detections:
            st.dataframe(detections, hide_index=True, use_container_width=True)
        else:
            st.write("Tidak ada objek yang terdeteksi pada threshold ini.")
        return

    file_suffix = Path(uploaded_file.name).suffix.lower()
    if file_suffix.lstrip(".") not in VIDEO_EXTENSIONS:
        st.error("Format video tidak didukung. Gunakan MP4, AVI, atau MOV.")
        return

    progress_bar = st.progress(0.0, text="Menyiapkan pemrosesan video...")
    try:
        with st.spinner("Memproses video frame-by-frame..."):
            video_data, statistics = process_video(
                model,
                uploaded_data,
                file_suffix,
                confidence,
                image_size,
                device,
                progress_bar,
            )
    except Exception as error:
        st.error(f"Video gagal diproses: {error}")
        return

    if statistics["fps_fallback_used"]:
        st.warning(
            "FPS video sumber tidak dapat dibaca; video hasil ditulis pada 30 FPS."
        )
    st.markdown("#### Video hasil deteksi")
    with st.container(width=600, horizontal_alignment="center"):
        st.video(video_data, format="video/mp4", width=600)

    st.markdown("#### Statistik video (akumulasi seluruh frame)")
    helmet_id, helmet_name = names_by_normalized["helmet"]
    no_helmet_id, no_helmet_name = names_by_normalized["no-helmet"]
    duration = float(statistics["duration"])
    metric_columns = st.columns(5)
    metric_columns[0].metric("Total frame", int(statistics["frames"]))
    metric_columns[1].metric(helmet_name, int(statistics["helmet"]))
    metric_columns[2].metric(no_helmet_name, int(statistics["no_helmet"]))
    metric_columns[3].metric(
        "Durasi",
        f"{duration:.2f} detik" if duration > 0 else "Tidak tersedia",
    )
    metric_columns[4].metric(
        "FPS / Resolusi",
        f"{float(statistics['fps']):g} / "
        f"{int(statistics['width'])}×{int(statistics['height'])}",
    )


if __name__ == "__main__":
    main()
