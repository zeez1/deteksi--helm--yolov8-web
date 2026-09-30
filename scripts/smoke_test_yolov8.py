#!/usr/bin/env python3
"""Run a short YOLOv8n training smoke test with all augmentation disabled."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "experiments" / "smoke_test"


def main() -> None:
    started = time.perf_counter()
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(ROOT / "dataset" / "helmet_research" / "data.yaml"),
        epochs=2,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        seed=42,
        project=str(PROJECT),
        name="yolov8n",
        exist_ok=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        cutmix=0.0,
        bgr=0.0,
        erasing=0.0,
        auto_augment=None,
        close_mosaic=0,
    )
    elapsed = time.perf_counter() - started
    print(f"TRAINING_METRICS: {results.results_dict}")
    print(f"TRAINING_SAVE_DIR: {results.save_dir}")
    print(f"TRAINING_SECONDS: {elapsed:.2f}")
    if torch.cuda.is_available():
        print(f"GPU_NAME: {torch.cuda.get_device_name(0)}")
        print(
            "GPU_PEAK_ALLOCATED_GIB: "
            f"{torch.cuda.max_memory_allocated(0) / 1024**3:.3f}"
        )
        print(
            "GPU_PEAK_RESERVED_GIB: "
            f"{torch.cuda.max_memory_reserved(0) / 1024**3:.3f}"
        )


if __name__ == "__main__":
    main()
