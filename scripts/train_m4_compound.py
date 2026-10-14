#!/usr/bin/env python3
"""Train M4 with the audited offline compound-augmentation dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
import ultralytics
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "dataset" / "experiments" / "M4_compound"
DATASET_CONFIG = DATASET_ROOT / "data.yaml"
AUDIT_PATH = ROOT / "reports" / "M4_dataset_audit.json"
PROJECT_DIR = ROOT / "experiments"
RUN_NAME = "M4_compound"
OUTPUT_DIR = PROJECT_DIR / RUN_NAME
REPORTS_DIR = ROOT / "reports"
CONFIG_PATH = REPORTS_DIR / "M4_config.json"
TEST_JSON_PATH = REPORTS_DIR / "M4_test_results.json"
TEST_TEXT_PATH = REPORTS_DIR / "M4_test_results.txt"
COMPARISON_PATH = REPORTS_DIR / "M0_M1_M2_M3_M4_comparison.csv"
SOURCE_ROOT = ROOT / "dataset" / "helmet_research"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_COUNTS = {"train": 635, "val": 181, "test": 92}
METRIC_KEYS = ("precision", "recall", "mAP50", "mAP50_95")
COMPARISON_METRICS = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("mAP50", "mAP50"),
    ("mAP50_95", "mAP50-95"),
)
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
EXPERIMENT_NAMES = {
    "M0": "baseline, no augmentation",
    "M1": "horizontal flip",
    "M2": "rotation 15 degrees",
    "M3": "brightness 0.8 + contrast 1.2",
    "M4": "horizontal flip + rotation + brightness + contrast",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
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


def validate_preflight() -> dict[str, int]:
    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(f"M4 dataset config not found: {DATASET_CONFIG}")
    dataset_config = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(dataset_config, dict):
        raise ValueError(f"M4 dataset config must be a mapping: {DATASET_CONFIG}")
    configured_root = Path(dataset_config.get("path", ""))
    if not configured_root.is_absolute():
        configured_root = (DATASET_CONFIG.parent / configured_root).resolve()
    if configured_root.resolve() != DATASET_ROOT.resolve():
        raise ValueError(f"M4 YAML points to unexpected dataset root: {configured_root}")
    if (
        dataset_config.get("train") != "train/images"
        or dataset_config.get("val") != "val/images"
        or dataset_config.get("test") != "test/images"
        or dataset_config.get("nc") != 2
        or dataset_config.get("names") != {0: "helmet", 1: "no-helmet"}
    ):
        raise ValueError(f"Unexpected M4 YAML configuration: {dataset_config}")

    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if (
        audit.get("experiment") != "M4_compound"
        or audit.get("passed") is not True
        or audit.get("error_count") != 0
    ):
        raise RuntimeError(f"M4 dataset audit did not pass: {AUDIT_PATH}")

    counts: dict[str, int] = {}
    for split, expected in SPLIT_COUNTS.items():
        split_root = DATASET_ROOT / split
        image_count = count_images(split_root / "images")
        label_count = sum(1 for path in (split_root / "labels").glob("*.txt") if path.is_file())
        if image_count != expected or label_count != expected:
            raise ValueError(
                f"Unexpected M4 {split} count: "
                f"{image_count} images, {label_count} labels"
            )
        counts[split] = image_count

    for split in ("val", "test"):
        source = SOURCE_ROOT / split
        target = DATASET_ROOT / split
        for source_file in source.rglob("*"):
            if not source_file.is_file():
                continue
            if source_file.parent.name not in {"images", "labels"}:
                continue
            if not (
                source_file.suffix.lower() in IMAGE_EXTENSIONS
                or source_file.suffix.lower() == ".txt"
            ):
                continue
            target_file = target / source_file.relative_to(source)
            if (
                not target_file.is_file()
                or sha256_file(source_file) != sha256_file(target_file)
            ):
                raise ValueError(
                    f"M4 {split} differs from source test/validation data: {source_file}"
                )

    model_weights = ROOT / "yolov8n.pt"
    if not model_weights.is_file():
        raise FileNotFoundError(f"Pretrained YOLOv8 weights not found: {model_weights}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for M4 training")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 2050":
        raise RuntimeError(
            "M4 requires NVIDIA GeForce RTX 2050, "
            f"found: {torch.cuda.get_device_name(0)}"
        )

    if OUTPUT_DIR.exists():
        if any(OUTPUT_DIR.iterdir()):
            raise FileExistsError(f"Refusing to overwrite M4 output: {OUTPUT_DIR}")
        raise FileExistsError(f"Refusing to reuse existing M4 output directory: {OUTPUT_DIR}")
    for path in (
        CONFIG_PATH,
        TEST_JSON_PATH,
        TEST_TEXT_PATH,
        COMPARISON_PATH,
    ):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    return counts


def load_test_metrics(name: str) -> dict[str, float]:
    report_path = REPORTS_DIR / f"{name}_test_results.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"{name} test report not found: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("split") != "test" or not isinstance(report.get("metrics"), dict):
        raise ValueError(f"Invalid {name} test report: {report_path}")
    metrics = report["metrics"]
    missing = [key for key in METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"{name} test report is missing metrics: {missing}")
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def metric_summary(metrics: Any) -> dict[str, Any]:
    box = metrics.box
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "inference_time_ms_per_image": float(metrics.speed["inference"]),
        "speed_ms_per_image": {
            key: float(value) for key, value in metrics.speed.items()
        },
    }


def write_comparison(all_metrics: dict[str, dict[str, float]]) -> None:
    baseline = all_metrics["M0"]
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "Model",
                "Augmentation",
                *(label for _, label in COMPARISON_METRICS),
                *(f"Delta_vs_M0_{label}" for _, label in COMPARISON_METRICS),
            ]
        )
        for name in ("M0", "M1", "M2", "M3", "M4"):
            metrics = all_metrics[name]
            writer.writerow(
                [
                    name,
                    EXPERIMENT_NAMES[name],
                    *(f"{metrics[key]:.9f}" for key, _ in COMPARISON_METRICS),
                    *(
                        f"{metrics[key] - baseline[key]:+.9f}"
                        for key, _ in COMPARISON_METRICS
                    ),
                ]
            )


def main() -> None:
    counts = validate_preflight()
    prior_metrics = {
        name: load_test_metrics(name) for name in ("M0", "M1", "M2", "M3")
    }
    model_path = ROOT / "yolov8n.pt"
    gpu_name = torch.cuda.get_device_name(0)
    config: dict[str, Any] = {
        "experiment": "M4_compound",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "model": str(model_path.resolve()),
        "pretrained": True,
        "dataset": str(DATASET_CONFIG.resolve()),
        "train_images": counts["train"],
        "validation_images": counts["val"],
        "test_images": counts["test"],
        "epochs": 50,
        "imgsz": 640,
        "batch": 4,
        "optimizer": "auto",
        "lr0": 0.01,
        "device": 0,
        "workers": 2,
        "seed": 42,
        "deterministic": True,
        "project": str(PROJECT_DIR.resolve()),
        "name": RUN_NAME,
        "training_uses_test_set": False,
        "offline_augmentation": {
            "sequence": [
                "horizontal_flip",
                "rotation",
                "brightness",
                "contrast",
            ],
            "rotation_degrees_range": [-15.0, 15.0],
            "rotation_seed": 42,
            "brightness_factor": 0.8,
            "contrast_factor": 1.2,
            "applied_to": "train only",
        },
        "augment": False,
        "auto_augment": None,
        "erasing": 0.0,
        "internal_augmentation": AUGMENTATION,
        "test_evaluation": "after training, on the M4 test split only",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CONFIG_PATH, config)
    OUTPUT_DIR.mkdir(parents=True)

    torch.cuda.reset_peak_memory_stats(0)
    training_started = time.perf_counter()
    model = YOLO(str(model_path))
    training_results = model.train(
        data=str(DATASET_CONFIG),
        epochs=50,
        imgsz=640,
        batch=4,
        optimizer="auto",
        lr0=0.01,
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
        auto_augment=None,
        erasing=0.0,
        close_mosaic=0,
        fraction=1.0,
        **{
            key: value
            for key, value in AUGMENTATION.items()
            if key not in {"auto_augment", "erasing"}
        },
    )
    training_seconds = time.perf_counter() - training_started
    result_dir = Path(training_results.save_dir)
    best_weights = result_dir / "weights" / "best.pt"
    last_weights = result_dir / "weights" / "last.pt"
    results_csv = result_dir / "results.csv"
    args_yaml = result_dir / "args.yaml"
    missing = [
        str(path)
        for path in (best_weights, last_weights, results_csv, args_yaml)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Required M4 training outputs missing: {missing}")

    with results_csv.open(newline="", encoding="utf-8-sig") as stream:
        training_rows = list(csv.DictReader(stream))
    if len(training_rows) != 50:
        raise RuntimeError(f"Expected 50 completed M4 epochs, found {len(training_rows)}")
    saved_args = yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
    expected_args = {
        "epochs": 50,
        "imgsz": 640,
        "batch": 4,
        "optimizer": "auto",
        "lr0": 0.01,
        "seed": 42,
        "device": "0",
        "deterministic": True,
        "augment": False,
        "auto_augment": None,
        "erasing": 0.0,
    }
    mismatched = {
        key: {"expected": value, "actual": saved_args.get(key)}
        for key, value in expected_args.items()
        if saved_args.get(key) != value
    }
    if mismatched:
        raise RuntimeError(f"M4 saved training arguments differ: {mismatched}")
    config.update(
        {
            "training_status": "completed",
            "training_seconds": training_seconds,
            "training_save_dir": str(result_dir.resolve()),
            "completed_epochs": len(training_rows),
            "training_metrics_last_epoch": training_rows[-1],
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }
    )

    best_model = YOLO(str(best_weights))
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
        "experiment": "M4_compound",
        "model": str(best_weights.resolve()),
        "dataset": str(DATASET_CONFIG.resolve()),
        "split": "test",
        "test_images": counts["test"],
        "metrics": test_summary,
        "training_seconds": training_seconds,
        "completed_epochs": len(training_rows),
        "gpu": gpu_name,
        "cuda_available": True,
        "gpu_peak_allocated_bytes": config["gpu_peak_allocated_bytes"],
        "gpu_peak_reserved_bytes": config["gpu_peak_reserved_bytes"],
        "cuda_out_of_memory": False,
    }
    all_metrics = {**prior_metrics, "M4": test_summary}
    write_comparison(all_metrics)
    write_json(TEST_JSON_PATH, test_report)
    config["test_results"] = test_summary
    write_json(CONFIG_PATH, config)

    report_lines = [
        "M4 COMPOUND — FINAL TEST EVALUATION",
        f"Model: {test_report['model']}",
        f"Dataset: {test_report['dataset']}",
        f"Split: test ({counts['test']} images)",
        f"Epochs: {len(training_rows)}",
        f"Training time: {training_seconds:.2f} seconds",
        f"Precision: {test_summary['precision']:.6f}",
        f"Recall: {test_summary['recall']:.6f}",
        f"mAP@50: {test_summary['mAP50']:.6f}",
        f"mAP@50-95: {test_summary['mAP50_95']:.6f}",
        "",
        "M0-M4 test metrics:",
        "Model,Precision,Recall,mAP50,mAP50-95",
    ]
    for name in ("M0", "M1", "M2", "M3", "M4"):
        metrics = all_metrics[name]
        report_lines.append(
            f"{name}," + ",".join(f"{metrics[key]:.9f}" for key in METRIC_KEYS)
        )
    report_lines.extend(
        [
            "",
            "Comparison CSV: " + str(COMPARISON_PATH.resolve()),
        ]
    )
    TEST_TEXT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("M4_TRAINING_COMPLETED: true")
    print(f"M4_COMPLETED_EPOCHS: {len(training_rows)}")
    print(f"M4_TRAINING_SECONDS: {training_seconds:.2f}")
    print(f"M4_BEST_PT: {best_weights.resolve()}")
    print(f"M4_LAST_PT: {last_weights.resolve()}")
    print(f"M4_RESULTS_CSV: {results_csv.resolve()}")
    print(f"M4_ARGS_YAML: {args_yaml.resolve()}")
    print(f"M4_CONFIG: {CONFIG_PATH.resolve()}")
    print(f"M4_TEST_JSON: {TEST_JSON_PATH.resolve()}")
    print(f"M4_TEST_TEXT: {TEST_TEXT_PATH.resolve()}")
    print(f"M0_M1_M2_M3_M4_COMPARISON: {COMPARISON_PATH.resolve()}")
    print("M4_TEST_METRICS: " + json.dumps(test_summary))
    for name in ("M0", "M1", "M2", "M3", "M4"):
        print(f"{name}_TEST_METRICS: " + json.dumps(all_metrics[name]))


if __name__ == "__main__":
    main()
