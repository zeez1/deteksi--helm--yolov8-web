#!/usr/bin/env python3
"""Train and evaluate M1 using the pre-flipped dataset and no online augmentation."""

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
DATASET_CONFIG = ROOT / "dataset" / "experiments" / "M1_flip" / "data.yaml"
PROJECT_DIR = ROOT / "experiments"
RUN_NAME = "M1_flip"
OUTPUT_DIR = PROJECT_DIR / RUN_NAME
REPORTS_DIR = ROOT / "reports"
CONFIG_PATH = REPORTS_DIR / "M1_config.json"
TEST_JSON_PATH = REPORTS_DIR / "M1_test_results.json"
TEST_TEXT_PATH = REPORTS_DIR / "M1_test_results.txt"
COMPARISON_PATH = REPORTS_DIR / "M0_vs_M1.csv"
M0_REPORT_PATH = REPORTS_DIR / "M0_test_results.json"
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
M0_METRICS = {
    "precision": 0.8171103675297509,
    "recall": 0.7080943461154818,
    "mAP50": 0.7662074843313005,
    "mAP50_95": 0.37796654814240205,
    "f1_score": 0.7587063247625684,
    "inference_time_ms_per_image": 8.364132609921914,
}


def count_images(directory: Path) -> int:
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    )


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_dataset() -> tuple[dict[str, Any], int, int, int]:
    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(
            f"M1 dataset config not found; create it first: {DATASET_CONFIG}"
        )
    config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = DATASET_CONFIG.parent / dataset_root
    counts = {
        split: count_images(dataset_root / split / "images")
        for split in ("train", "val", "test")
    }
    expected = {"train": 635, "val": 181, "test": 92}
    if counts != expected:
        raise ValueError(f"Unexpected M1 dataset counts: {counts}; expected {expected}")
    return config, counts["train"], counts["val"], counts["test"]


def write_comparison(
    m0_metrics: dict[str, Any], m1_metrics: dict[str, Any]
) -> None:
    metric_columns = (
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "f1_score",
        "inference_time_ms_per_image",
    )
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        metric_headers = (
            "Precision",
            "Recall",
            "mAP50",
            "mAP50-95",
            "F1",
            "Inference_Time_ms",
        )
        writer.writerow(
            [
                "Model",
                *metric_headers,
                *(f"Delta_vs_M0_{header}" for header in metric_headers),
            ]
        )
        for model_name, metrics in (("M0", m0_metrics), ("M1", m1_metrics)):
            baseline = m0_metrics
            writer.writerow(
                [
                    model_name,
                    *(f"{metrics[key]:.9f}" for key in metric_columns),
                    *(
                        f"{metrics[key] - baseline[key]:+.9f}"
                        for key in metric_columns
                    ),
                ]
            )


def train_and_evaluate() -> None:
    _, train_count, val_count, test_count = validate_dataset()
    audit_path = ROOT / "reports" / "M1_dataset_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"M1 dataset audit not found: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("error_count") != 0:
        raise RuntimeError(f"M1 dataset audit has not passed: {audit_path}")
    if not M0_REPORT_PATH.is_file():
        raise FileNotFoundError(f"M0 test metrics not found: {M0_REPORT_PATH}")
    m0_report = json.loads(M0_REPORT_PATH.read_text(encoding="utf-8"))
    if m0_report.get("split") != "test" or not isinstance(
        m0_report.get("metrics"), dict
    ):
        raise ValueError(f"Invalid M0 test metrics report: {M0_REPORT_PATH}")
    m0_metrics = m0_report["metrics"]
    for key in M0_METRICS:
        if key not in m0_metrics:
            raise ValueError(f"M0 test report is missing metric {key!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for M1; no CUDA device is available")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 2050":
        raise RuntimeError(
            "M1 must use NVIDIA GeForce RTX 2050, found: "
            f"{torch.cuda.get_device_name(0)}"
        )
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite M1 output: {OUTPUT_DIR}")
    for report_path in (CONFIG_PATH, TEST_JSON_PATH, TEST_TEXT_PATH, COMPARISON_PATH):
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu_name = torch.cuda.get_device_name(0)
    config: dict[str, Any] = {
        "experiment": "M1_flip",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "model": "yolov8n.pt",
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
        "horizontal_flip": {"probability": 1.0, "applied_before_training": True},
        "augment": False,
        "internal_augmentation": AUGMENTATION,
        "test_evaluation": "after training, on the test split only",
    }
    write_json(CONFIG_PATH, config)

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
    last_weights = result_dir / "weights" / "last.pt"
    results_csv = result_dir / "results.csv"
    required_files = (
        best_weights,
        last_weights,
        results_csv,
        result_dir / "results.png",
        result_dir / "args.yaml",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required M1 training outputs missing: {missing}")

    with results_csv.open(newline="", encoding="utf-8-sig") as stream:
        training_rows = list(csv.DictReader(stream))
    if len(training_rows) != 50:
        raise RuntimeError(f"Expected 50 completed epochs; found {len(training_rows)}")
    config["training_seconds"] = training_seconds
    config["training_save_dir"] = str(result_dir.resolve())
    config["completed_epochs"] = len(training_rows)
    config["training_metrics_last_epoch"] = training_rows[-1]
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
        "experiment": "M1_flip",
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

    confusion_source = Path(validation_metrics.save_dir) / "confusion_matrix.png"
    if confusion_source.is_file():
        shutil.copy2(confusion_source, result_dir / "confusion_matrix.png")

    confusion_matrix = result_dir / "confusion_matrix.png"
    if not confusion_matrix.is_file():
        raise FileNotFoundError(f"Required M1 confusion matrix missing: {confusion_matrix}")

    write_json(TEST_JSON_PATH, test_report)
    write_comparison(m0_metrics, test_summary)
    report_lines = [
        "M1 HORIZONTAL FLIP — FINAL TEST EVALUATION",
        f"Model: {test_report['model']}",
        f"Dataset: {test_report['dataset']}",
        f"Split: {test_report['split']} ({test_count} images)",
        f"Training images: {train_count} (each replaced by its horizontal flip)",
        f"Validation images: {val_count}",
        "Horizontal flip probability: 1.0",
        "Internal YOLO augmentation: disabled",
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
        "",
        "M0 vs M1 (test metrics; delta is M1 - M0):",
        "Model,Precision,Recall,mAP50,mAP50-95,F1,Inference_Time_ms",
    ]
    for model_name, metrics in (("M0", m0_metrics), ("M1", test_summary)):
        report_lines.append(
            f"{model_name},"
            + ",".join(f"{metrics[key]:.9f}" for key in M0_METRICS)
        )
    report_lines.append(
        "M1-M0,"
        + ",".join(
            f"{test_summary[key] - m0_metrics[key]:+.9f}" for key in M0_METRICS
        )
    )
    TEST_TEXT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(CONFIG_PATH, config)

    print("M1_TRAINING_COMPLETED: true")
    print(f"M1_OUTPUT_DIR: {result_dir.resolve()}")
    print(f"M1_BEST_PT: {best_weights.resolve()}")
    print(f"M1_LAST_PT: {last_weights.resolve()}")
    print(f"M1_RESULTS_CSV: {results_csv.resolve()}")
    print(f"M1_RESULTS_PNG: {(result_dir / 'results.png').resolve()}")
    print(f"M1_CONFUSION_MATRIX: {(result_dir / 'confusion_matrix.png').resolve()}")
    print(f"M1_CONFIG: {CONFIG_PATH.resolve()}")
    print(f"M1_VALIDATION_METRICS: {json.dumps(validation_summary)}")
    print(f"M1_TEST_METRICS: {json.dumps(test_summary)}")
    print(f"M1_TEST_REPORT_JSON: {TEST_JSON_PATH.resolve()}")
    print(f"M1_TEST_REPORT_TXT: {TEST_TEXT_PATH.resolve()}")
    print(f"M0_VS_M1_COMPARISON: {COMPARISON_PATH.resolve()}")


if __name__ == "__main__":
    train_and_evaluate()
