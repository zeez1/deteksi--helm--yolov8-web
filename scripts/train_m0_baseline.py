#!/usr/bin/env python3
"""Train M0 without augmentation, then evaluate validation and test once."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG = ROOT / "dataset" / "helmet_research" / "data.yaml"
PROJECT_DIR = ROOT / "experiments"
RUN_NAME = "M0_baseline"
OUTPUT_DIR = PROJECT_DIR / RUN_NAME
REPORTS_DIR = ROOT / "reports"
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
AUGMENTATION = {
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "fliplr": 0.0,
    "flipud": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "cutmix": 0.0,
    "bgr": 0.0,
    "erasing": 0.0,
    "auto_augment": None,
    "multi_scale": 0.0,
}


def count_test_images(dataset_yaml: Path) -> int:
    with dataset_yaml.open(encoding="utf-8") as stream:
        dataset_config = yaml.safe_load(stream)
    if not isinstance(dataset_config, dict):
        raise ValueError(f"Dataset config must contain a YAML mapping: {dataset_yaml}")

    configured_root = dataset_config.get("path")
    if configured_root is None:
        dataset_root = dataset_yaml.parent
    else:
        dataset_root = Path(configured_root)
        if not dataset_root.is_absolute():
            dataset_root = dataset_yaml.parent / dataset_root
    test_entry = dataset_config.get("test")
    if not test_entry:
        raise ValueError(f"Dataset config has no test split: {dataset_yaml}")
    entries = test_entry if isinstance(test_entry, list) else [test_entry]

    image_paths: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"Invalid test split entry in {dataset_yaml}: {entry!r}")
        path = Path(entry)
        if not path.is_absolute():
            path = dataset_root / path
        path = path.resolve()
        if path.is_dir():
            image_paths.update(
                candidate.resolve()
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
            )
        elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.add(path)
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                item = line.strip()
                if not item:
                    continue
                image_path = Path(item)
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                image_path = image_path.resolve()
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    raise FileNotFoundError(f"Test image listed in {path} is missing: {image_path}")
                image_paths.add(image_path)
        else:
            raise FileNotFoundError(f"Test split path is missing or unsupported: {path}")

    if not image_paths:
        raise ValueError(f"No test images found from dataset config: {dataset_yaml}")
    return len(image_paths)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def metric_summary(metrics: Any) -> dict[str, Any]:
    box = metrics.box
    precision = float(box.mp)
    recall = float(box.mr)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    speed_ms = {
        key: float(value)
        for key, value in metrics.speed.items()
    }
    inference_ms = speed_ms.get("inference")
    return {
        "precision": precision,
        "recall": recall,
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "f1_score": f1,
        "inference_time_ms_per_image": inference_ms,
        "speed_ms_per_image": speed_ms,
    }


def train_and_evaluate() -> None:
    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(f"Dataset config not found: {DATASET_CONFIG}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite M0 output: {OUTPUT_DIR}")
    test_image_count = count_test_images(DATASET_CONFIG)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    config: dict[str, Any] = {
        "experiment": "M0_baseline",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name,
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None
        ),
        "model": "yolov8n.pt",
        "dataset": str(DATASET_CONFIG.resolve()),
        "epochs": 50,
        "imgsz": 640,
        "batch": 4,
        "device": 0,
        "workers": 2,
        "seed": 42,
        "project": str(PROJECT_DIR.resolve()),
        "name": RUN_NAME,
        "test_image_count": test_image_count,
        "augmentation": AUGMENTATION,
        "training_uses_test_set": False,
        "test_evaluation": "after training, on the test split only",
    }
    config_path = REPORTS_DIR / "M0_config.json"
    write_json(config_path, config)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    training_started = time.perf_counter()
    model = YOLO("yolov8n.pt")
    training_results = model.train(
        data=str(DATASET_CONFIG),
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        seed=42,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        val=True,
        save=True,
        plots=True,
        augment=False,
        close_mosaic=0,
        **AUGMENTATION,
    )
    training_seconds = time.perf_counter() - training_started
    result_dir = Path(training_results.save_dir)
    config["training_seconds"] = training_seconds
    config["training_save_dir"] = str(result_dir.resolve())

    best_weights = result_dir / "weights" / "best.pt"
    last_weights = result_dir / "weights" / "last.pt"
    results_csv = result_dir / "results.csv"
    required_files = (best_weights, last_weights, results_csv)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        write_json(config_path, config)
        raise FileNotFoundError(f"Required M0 training outputs missing: {missing}")

    training_rows: list[dict[str, str]] = []
    with results_csv.open(newline="", encoding="utf-8-sig") as stream:
        training_rows = list(csv.DictReader(stream))
    config["completed_epochs"] = len(training_rows)
    config["training_metrics_last_epoch"] = training_rows[-1] if training_rows else {}
    if torch.cuda.is_available():
        config["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
        config["gpu_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)

    best_model = YOLO(str(best_weights))
    validation_metrics = best_model.val(
        data=str(DATASET_CONFIG),
        split="val",
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project=str(OUTPUT_DIR / "evaluation"),
        name="validation",
        exist_ok=True,
        plots=True,
        save_json=False,
    )
    validation_summary = metric_summary(validation_metrics)
    config["validation_results"] = validation_summary

    test_metrics = best_model.val(
        data=str(DATASET_CONFIG),
        split="test",
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project=str(OUTPUT_DIR / "evaluation"),
        name="test",
        exist_ok=True,
        plots=True,
        save_json=False,
    )
    test_summary = metric_summary(test_metrics)
    test_report = {
        "experiment": "M0_baseline",
        "model": str(best_weights.resolve()),
        "dataset": str(DATASET_CONFIG.resolve()),
        "split": "test",
        "test_images": test_image_count,
        "metrics": test_summary,
        "training_seconds": training_seconds,
        "gpu": gpu_name,
        "cuda_available": torch.cuda.is_available(),
        "gpu_peak_allocated_bytes": config.get("gpu_peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": config.get("gpu_peak_reserved_bytes"),
        "cuda_out_of_memory": False,
    }
    test_json = REPORTS_DIR / "M0_test_results.json"
    write_json(test_json, test_report)
    report_lines = [
        "M0 BASELINE — FINAL TEST EVALUATION",
        f"Model: {test_report['model']}",
        f"Dataset: {test_report['dataset']}",
        f"Split: {test_report['split']} ({test_report['test_images']} images)",
        f"Precision: {test_summary['precision']:.6f}",
        f"Recall: {test_summary['recall']:.6f}",
        f"mAP@50: {test_summary['mAP50']:.6f}",
        f"mAP@50-95: {test_summary['mAP50_95']:.6f}",
        f"F1-score: {test_summary['f1_score']:.6f}",
        (
            "Inference time: "
            f"{test_summary['inference_time_ms_per_image']:.3f} ms/image"
            if test_summary["inference_time_ms_per_image"] is not None
            else "Inference time: unavailable"
        ),
        f"Training time: {training_seconds:.2f} seconds",
        f"GPU: {gpu_name}",
        "CUDA OOM: no",
    ]
    (REPORTS_DIR / "M0_test_results.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    write_json(config_path, config)

    confusion_source = Path(validation_metrics.save_dir) / "confusion_matrix.png"
    if confusion_source.is_file():
        shutil.copy2(confusion_source, result_dir / "confusion_matrix.png")

    print("M0_TRAINING_COMPLETED: true")
    print(f"M0_OUTPUT_DIR: {result_dir.resolve()}")
    print(f"M0_BEST_PT: {best_weights.resolve()}")
    print(f"M0_LAST_PT: {last_weights.resolve()}")
    print(f"M0_RESULTS_CSV: {results_csv.resolve()}")
    print(f"M0_RESULTS_PNG: {(result_dir / 'results.png').resolve()}")
    print(f"M0_CONFUSION_MATRIX: {(result_dir / 'confusion_matrix.png').resolve()}")
    print(f"M0_TRAINING_SECONDS: {training_seconds:.2f}")
    print(f"M0_VALIDATION_METRICS: {json.dumps(validation_summary)}")
    print(f"M0_TEST_METRICS: {json.dumps(test_summary)}")
    print(f"M0_TEST_REPORT_JSON: {test_json.resolve()}")
    print(f"M0_TEST_REPORT_TXT: {(REPORTS_DIR / 'M0_test_results.txt').resolve()}")
    print(f"M0_CONFIG: {config_path.resolve()}")
    if torch.cuda.is_available():
        print(f"M0_GPU: {gpu_name}")
        print(
            "M0_GPU_PEAK_ALLOCATED_GIB: "
            f"{torch.cuda.max_memory_allocated(0) / 1024**3:.3f}"
        )
        print(
            "M0_GPU_PEAK_RESERVED_GIB: "
            f"{torch.cuda.max_memory_reserved(0) / 1024**3:.3f}"
        )


if __name__ == "__main__":
    train_and_evaluate()
