#!/usr/bin/env python3
"""Train M3 on the brightness/contrast dataset and evaluate val and test."""

from __future__ import annotations

import csv
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG = (
    ROOT / "dataset" / "experiments" / "M3_brightness_contrast" / "data.yaml"
)
AUDIT_PATH = ROOT / "reports" / "M3_dataset_audit.json"
PROJECT_DIR = ROOT / "experiments"
RUN_NAME = "M3_brightness_contrast"
OUTPUT_DIR = PROJECT_DIR / RUN_NAME
REPORTS_DIR = ROOT / "reports"
CONFIG_PATH = REPORTS_DIR / "M3_config.json"
TEST_JSON_PATH = REPORTS_DIR / "M3_test_results.json"
TEST_TEXT_PATH = REPORTS_DIR / "M3_test_results.txt"
COMPARISON_PATH = REPORTS_DIR / "M0_M1_M2_M3_comparison.csv"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
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
    "auto_augment": None,
    "erasing": 0.0,
}
METRIC_KEYS = (
    "precision",
    "recall",
    "mAP50",
    "mAP50_95",
    "f1_score",
    "inference_time_ms_per_image",
)
METRIC_COLUMNS = (
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95",
    "F1",
    "Inference_Time_ms",
)
MODEL_AUGMENTATIONS = {
    "M0": "none",
    "M1": "horizontal flip",
    "M2": "rotation",
    "M3": "brightness + contrast",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def count_images(directory: Path) -> int:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset image directory not found: {directory}")
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset() -> tuple[int, int, int]:
    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(f"M3 dataset config not found: {DATASET_CONFIG}")
    dataset = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise ValueError(f"Dataset config must be a YAML mapping: {DATASET_CONFIG}")
    dataset_root = Path(dataset["path"])
    if not dataset_root.is_absolute():
        dataset_root = DATASET_CONFIG.parent / dataset_root
    counts = {
        split: count_images(dataset_root / split / "images")
        for split in ("train", "val", "test")
    }
    expected = {"train": 635, "val": 181, "test": 92}
    if counts != expected:
        raise ValueError(f"Unexpected M3 split counts: {counts}; expected {expected}")
    return counts["train"], counts["val"], counts["test"]


def metric_summary(metrics: Any) -> dict[str, Any]:
    box = metrics.box
    precision = float(box.mp)
    recall = float(box.mr)
    speed_ms = {key: float(value) for key, value in metrics.speed.items()}
    return {
        "precision": precision,
        "recall": recall,
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "f1_score": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
        "inference_time_ms_per_image": speed_ms.get("inference"),
        "speed_ms_per_image": speed_ms,
    }


def load_test_metrics(path: Path, model_name: str) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("split") != "test" or not isinstance(
        report.get("metrics"), dict
    ):
        raise ValueError(f"Invalid {model_name} test report: {path}")
    metrics = report["metrics"]
    missing = [key for key in METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"{model_name} report is missing metrics: {missing}")
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def write_comparison(test_metrics: dict[str, dict[str, float]]) -> None:
    delta_columns = tuple(f"Delta_M3_vs_M0_{name}" for name in METRIC_COLUMNS)
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["Model", "Augmentation", *METRIC_COLUMNS, *delta_columns]
        )
        baseline = test_metrics["M0"]
        for model_name in ("M0", "M1", "M2", "M3"):
            metrics = test_metrics[model_name]
            deltas = (
                [
                    f"{metrics[key] - baseline[key]:+.9f}"
                    for key in METRIC_KEYS
                ]
                if model_name == "M3"
                else [""] * len(METRIC_KEYS)
            )
            writer.writerow(
                [
                    model_name,
                    MODEL_AUGMENTATIONS[model_name],
                    *(f"{metrics[key]:.9f}" for key in METRIC_KEYS),
                    *deltas,
                ]
            )


def main() -> None:
    train_count, val_count, test_count = validate_dataset()
    if not AUDIT_PATH.is_file():
        raise FileNotFoundError(f"M3 dataset audit not found: {AUDIT_PATH}")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if (
        audit.get("experiment") != "M3_brightness_contrast"
        or not audit.get("passed")
        or audit.get("error_count") != 0
    ):
        raise RuntimeError(f"M3 dataset audit did not pass: {AUDIT_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for M3; no CUDA device is available")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != "NVIDIA GeForce RTX 2050":
        raise RuntimeError(f"M3 requires NVIDIA GeForce RTX 2050, found: {gpu_name}")
    model_path = ROOT / "yolov8n.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Required model weights not found: {model_path}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite M3 output: {OUTPUT_DIR}")
    for path in (
        CONFIG_PATH,
        TEST_JSON_PATH,
        TEST_TEXT_PATH,
        COMPARISON_PATH,
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    test_metrics = {
        name: load_test_metrics(REPORTS_DIR / f"{name}_test_results.json", name)
        for name in ("M0", "M1", "M2")
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "experiment": "M3_brightness_contrast",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "model": str(model_path.resolve()),
        "dataset": str(DATASET_CONFIG.resolve()),
        "train_images": train_count,
        "validation_images": val_count,
        "test_images": test_count,
        "epochs": 50,
        "imgsz": 640,
        "batch": 4,
        "device": 0,
        "workers": 2,
        "seed": 42,
        "project": str(PROJECT_DIR.resolve()),
        "name": RUN_NAME,
        "training_uses_test_set": False,
        "offline_augmentation": {
            "type": "brightness + contrast",
            "brightness_factor": 0.8,
            "contrast_factor": 1.2,
            "seed": 42,
            "applied_to": "train only",
            "original_train_images_included": False,
            "horizontal_flip": False,
            "rotation": False,
            "scale": False,
            "other_augmentation": False,
            "bounding_boxes_transformed": False,
        },
        "augment": False,
        "internal_augmentation": AUGMENTATION,
        "test_evaluation": "after training and validation evaluation, on the test split only",
    }
    write_json(CONFIG_PATH, config)

    torch.cuda.reset_peak_memory_stats(0)
    training_started = time.perf_counter()
    try:
        model = YOLO(str(model_path))
        training_results = model.train(
            data=str(DATASET_CONFIG),
            epochs=50,
            imgsz=640,
            batch=4,
            device=0,
            workers=2,
            seed=42,
            deterministic=True,
            project=str(PROJECT_DIR),
            name=RUN_NAME,
            exist_ok=True,
            pretrained=True,
            val=True,
            save=True,
            plots=True,
            augment=False,
            close_mosaic=0,
            fraction=1.0,
            **AUGMENTATION,
        )
        training_seconds = time.perf_counter() - training_started
        result_dir = Path(training_results.save_dir)
        best_weights = result_dir / "weights" / "best.pt"
        required_files = (
            best_weights,
            result_dir / "weights" / "last.pt",
            result_dir / "results.csv",
            result_dir / "results.png",
            result_dir / "args.yaml",
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Required M3 training outputs missing: {missing}")

        with (result_dir / "results.csv").open(
            newline="", encoding="utf-8-sig"
        ) as stream:
            training_rows = list(csv.DictReader(stream))
        if len(training_rows) != 50:
            raise RuntimeError(f"Expected 50 completed epochs; found {len(training_rows)}")
        config.update(
            {
                "training_seconds": training_seconds,
                "training_save_dir": str(result_dir.resolve()),
                "completed_epochs": len(training_rows),
                "training_metrics_last_epoch": training_rows[-1],
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            }
        )

        best_model = YOLO(str(best_weights))
        validation_result = best_model.val(
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
        validation_summary = metric_summary(validation_result)
        config["validation_results"] = validation_summary

        test_result = best_model.val(
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
        test_summary = metric_summary(test_result)
        config["test_results"] = test_summary

        confusion_source = Path(validation_result.save_dir) / "confusion_matrix.png"
        if not confusion_source.is_file():
            raise FileNotFoundError(
                f"Validation confusion matrix not found: {confusion_source}"
            )
        shutil.copy2(confusion_source, result_dir / "confusion_matrix.png")
        write_json(CONFIG_PATH, config)

        test_metrics["M3"] = test_summary
        test_report = {
            "experiment": "M3_brightness_contrast",
            "model": str(best_weights.resolve()),
            "dataset": str(DATASET_CONFIG.resolve()),
            "split": "test",
            "test_images": test_count,
            "metrics": test_summary,
            "training_seconds": training_seconds,
            "gpu": gpu_name,
            "cuda_available": True,
            "gpu_peak_allocated_bytes": config["gpu_peak_allocated_bytes"],
            "gpu_peak_reserved_bytes": config["gpu_peak_reserved_bytes"],
            "cuda_out_of_memory": False,
        }
        write_json(TEST_JSON_PATH, test_report)
        write_comparison(test_metrics)
        report_lines = [
            "M3 BRIGHTNESS + CONTRAST — FINAL TEST EVALUATION",
            f"Model: {test_report['model']}",
            f"Dataset: {test_report['dataset']}",
            f"Split: test ({test_count} images)",
            f"Training images: {train_count} (one augmented image per source image)",
            f"Validation images: {val_count}",
            "Brightness factor: 0.8",
            "Contrast factor: 1.2",
            "Seed: 42",
            "Horizontal flip, rotation, scale, and other augmentation: disabled",
            "Internal YOLO augmentation: disabled",
            f"Precision: {test_summary['precision']:.6f}",
            f"Recall: {test_summary['recall']:.6f}",
            f"mAP@50: {test_summary['mAP50']:.6f}",
            f"mAP@50-95: {test_summary['mAP50_95']:.6f}",
            f"F1-score: {test_summary['f1_score']:.6f}",
            (
                f"Inference time: {test_summary['inference_time_ms_per_image']:.3f} ms/image"
                if test_summary["inference_time_ms_per_image"] is not None
                else "Inference time: unavailable"
            ),
            f"Training time: {training_seconds:.2f} seconds",
            f"GPU: {gpu_name}",
            "",
            "Validation metrics:",
            f"Precision: {validation_summary['precision']:.6f}",
            f"Recall: {validation_summary['recall']:.6f}",
            f"mAP@50: {validation_summary['mAP50']:.6f}",
            f"mAP@50-95: {validation_summary['mAP50_95']:.6f}",
            f"F1-score: {validation_summary['f1_score']:.6f}",
            (
                f"Inference time: {validation_summary['inference_time_ms_per_image']:.3f} ms/image"
                if validation_summary["inference_time_ms_per_image"] is not None
                else "Inference time: unavailable"
            ),
        ]
        TEST_TEXT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        print(f"M3_CONFIG: {CONFIG_PATH.resolve()}")
        print(f"M3_TEST_RESULTS_JSON: {TEST_JSON_PATH.resolve()}")
        print(f"M3_TEST_RESULTS_TEXT: {TEST_TEXT_PATH.resolve()}")
        print(f"M3_COMPARISON: {COMPARISON_PATH.resolve()}")
        print(f"M3_OUTPUT: {result_dir.resolve()}")
        print("VALIDATION_METRICS: " + json.dumps(validation_summary))
        print("TEST_METRICS: " + json.dumps(test_summary))
    except torch.cuda.OutOfMemoryError:
        config["cuda_out_of_memory"] = True
        config["training_seconds"] = time.perf_counter() - training_started
        write_json(CONFIG_PATH, config)
        raise


if __name__ == "__main__":
    main()
