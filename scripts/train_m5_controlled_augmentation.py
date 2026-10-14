#!/usr/bin/env python3
"""Preflight or train M5 with controlled online detection augmentations."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.data.augment import Albumentations, v8_transforms
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import IterableSimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG = ROOT / "dataset" / "helmet_research" / "data.yaml"
DATASET_ROOT = ROOT / "dataset" / "helmet_research"
MODEL_PATH = ROOT / "yolov8n.pt"
PROJECT_DIR = ROOT / "experiments"
RUN_NAME = "M5_controlled_augmentation"
OUTPUT_DIR = PROJECT_DIR / RUN_NAME
REPORTS_DIR = ROOT / "reports"
CONFIG_PATH = REPORTS_DIR / "M5_config.json"
TEST_JSON_PATH = REPORTS_DIR / "M5_test_results.json"
TEST_TEXT_PATH = REPORTS_DIR / "M5_test_results.txt"
COMPARISON_PATH = REPORTS_DIR / "M0_M1_M2_M3_M4_M5_comparison.csv"
REQUIRED_ULTRALYTICS_VERSION = "8.4.165"
REQUIRED_ALBUMENTATIONS_VERSION = "1.4.24"
SPLIT_COUNTS = {"train": 635, "val": 181, "test": 92}
METRIC_KEYS = ("precision", "recall", "mAP50", "mAP50_95")
METRIC_COLUMNS = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("mAP50", "mAP50"),
    ("mAP50_95", "mAP50-95"),
)
EXPERIMENT_NAMES = {
    "M0": "baseline, no augmentation",
    "M1": "horizontal flip",
    "M2": "rotation 15 degrees",
    "M3": "brightness 0.8 + contrast 1.2",
    "M4": "horizontal flip + rotation + brightness + contrast",
    "M5": "online flip + light rotation/scale/translation + brightness/contrast",
}
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

TRAIN_ARGUMENTS: dict[str, Any] = {
    "epochs": 50,
    "imgsz": 640,
    "batch": 4,
    "optimizer": "auto",
    "lr0": 0.01,
    "device": 0,
    "workers": 2,
    "seed": 42,
    "deterministic": True,
    "pretrained": True,
    "val": True,
    "save": True,
    "plots": True,
    "augment": False,
    "auto_augment": None,
    "erasing": 0.0,
    "close_mosaic": 0,
    "fraction": 1.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cutmix": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "degrees": 5.0,
    "translate": 0.05,
    "scale": 0.1,
    "shear": 0.0,
    "perspective": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "bgr": 0.0,
    "multi_scale": 0.0,
}

BRIGHTNESS_CONTRAST = {
    "transform": "albumentations.RandomBrightnessContrast",
    "brightness_limit": 0.1,
    "contrast_limit": 0.1,
    "probability": 0.2,
}


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_worktree() -> None:
    git_root = Path(git_output("rev-parse", "--show-toplevel")).resolve()
    if git_root != ROOT.resolve():
        raise RuntimeError(f"Unexpected Git worktree root: {git_root}")
    if git_output("branch", "--show-current") != "master":
        raise RuntimeError("M5 must run from the main worktree on branch master.")


def image_count(directory: Path) -> int:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset image directory not found: {directory}")
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset() -> dict[str, int]:
    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(f"Original dataset YAML not found: {DATASET_CONFIG}")
    dataset = yaml.safe_load(DATASET_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise ValueError(f"Dataset YAML must be a mapping: {DATASET_CONFIG}")

    configured_root = Path(dataset.get("path", DATASET_CONFIG.parent))
    if not configured_root.is_absolute():
        configured_root = DATASET_CONFIG.parent / configured_root
    if configured_root.resolve() != DATASET_ROOT.resolve():
        raise ValueError(
            f"M5 must use the original M0-M3 dataset, not {configured_root}"
        )
    expected_entries = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
    }
    for key, value in expected_entries.items():
        if dataset.get(key) != value:
            raise ValueError(f"Unexpected {key} dataset entry: {dataset.get(key)!r}")
    if dataset.get("nc") != 2 or dataset.get("names") != {
        0: "helmet",
        1: "no-helmet",
    }:
        raise ValueError(f"Unexpected class configuration: {dataset}")

    counts: dict[str, int] = {}
    for split, expected in SPLIT_COUNTS.items():
        split_root = DATASET_ROOT / split
        images = image_count(split_root / "images")
        labels = sum(1 for path in (split_root / "labels").glob("*.txt") if path.is_file())
        if images != expected or labels != expected:
            raise ValueError(
                f"Unexpected {split} counts: {images} images and {labels} labels; "
                f"expected {expected} of each"
            )
        counts[split] = images
    return counts


def validate_dependency_and_api() -> None:
    if ultralytics.__version__ != REQUIRED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Expected Ultralytics {REQUIRED_ULTRALYTICS_VERSION}, "
            f"found {ultralytics.__version__}"
        )
    if A.__version__ != REQUIRED_ALBUMENTATIONS_VERSION:
        raise RuntimeError(
            f"Expected Albumentations {REQUIRED_ALBUMENTATIONS_VERSION}, "
            f"found {A.__version__}"
        )
    signature = inspect.signature(A.RandomBrightnessContrast)
    for parameter in ("brightness_limit", "contrast_limit", "p"):
        if parameter not in signature.parameters:
            raise RuntimeError(
                "Installed RandomBrightnessContrast API is missing "
                f"the {parameter!r} parameter"
            )
    if "trainer" not in inspect.signature(YOLO.train).parameters:
        raise RuntimeError("YOLO.train does not support the required custom trainer API")

    custom_transform = A.RandomBrightnessContrast(
        brightness_limit=BRIGHTNESS_CONTRAST["brightness_limit"],
        contrast_limit=BRIGHTNESS_CONTRAST["contrast_limit"],
        p=BRIGHTNESS_CONTRAST["probability"],
    )
    hyp = IterableSimpleNamespace(**vars(DEFAULT_CFG))
    hyp.augmentations = [custom_transform]
    dataset_stub = SimpleNamespace(
        use_obb=False,
        use_keypoints=False,
        cache=False,
        data={"names": {0: "helmet", 1: "no-helmet"}},
    )
    transforms = v8_transforms(dataset_stub, imgsz=640, hyp=hyp)
    albumentations_transforms = [
        transform
        for transform in transforms.transforms
        if isinstance(transform, Albumentations)
    ]
    if (
        len(albumentations_transforms) != 1
        or albumentations_transforms[0].transform is None
        or not any(
            isinstance(transform, A.RandomBrightnessContrast)
            for transform in albumentations_transforms[0].transform.transforms
        )
    ):
        raise RuntimeError(
            "Brightness/contrast did not register in the Ultralytics detection "
            "training transform pipeline"
        )


class M5DetectionTrainer(DetectionTrainer):
    """Inject the validated Albumentations operation into train data only."""

    def build_dataset(
        self, img_path: str, mode: str = "train", batch: int | None = None
    ):
        if mode == "train":
            self.args.augmentations = [
                A.RandomBrightnessContrast(
                    brightness_limit=BRIGHTNESS_CONTRAST["brightness_limit"],
                    contrast_limit=BRIGHTNESS_CONTRAST["contrast_limit"],
                    p=BRIGHTNESS_CONTRAST["probability"],
                )
            ]
        return super().build_dataset(img_path, mode=mode, batch=batch)


def check_output_paths(*, allow_pending: bool) -> bool:
    report_paths = (CONFIG_PATH, TEST_JSON_PATH, TEST_TEXT_PATH, COMPARISON_PATH)
    present = [path for path in report_paths if path.exists()]
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to reuse or overwrite M5 experiment output: {OUTPUT_DIR}"
        )
    if not present:
        return False
    if not allow_pending or len(present) != len(report_paths):
        raise FileExistsError(f"Refusing to overwrite existing M5 reports: {present}")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result = json.loads(TEST_JSON_PATH.read_text(encoding="utf-8"))
    if (
        config.get("experiment") != RUN_NAME
        or config.get("preflight_status") != "passed"
        or result.get("experiment") != RUN_NAME
        or result.get("status") != "pending_training"
    ):
        raise RuntimeError("Existing M5 reports are not this script's pending outputs")
    return True


def load_test_metrics(name: str) -> dict[str, float]:
    path = REPORTS_DIR / f"{name}_test_results.json"
    if not path.is_file():
        raise FileNotFoundError(f"{name} test report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report.get("metrics")
    if report.get("split") != "test" or not isinstance(metrics, dict):
        raise ValueError(f"Invalid {name} test report: {path}")
    missing = [key for key in METRIC_KEYS if key not in metrics]
    if missing:
        raise ValueError(f"{name} test report is missing metrics: {missing}")
    return {key: float(metrics[key]) for key in METRIC_KEYS}


def base_config(counts: dict[str, int]) -> dict[str, Any]:
    return {
        "experiment": RUN_NAME,
        "preflight_status": "passed",
        "training_status": "not_started",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "albumentations_version": A.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "model": str(MODEL_PATH.resolve()),
        "pretrained": True,
        "dataset": str(DATASET_CONFIG.resolve()),
        "train_images": counts["train"],
        "validation_images": counts["val"],
        "test_images": counts["test"],
        "epochs": TRAIN_ARGUMENTS["epochs"],
        "imgsz": TRAIN_ARGUMENTS["imgsz"],
        "batch": TRAIN_ARGUMENTS["batch"],
        "optimizer": TRAIN_ARGUMENTS["optimizer"],
        "lr0": TRAIN_ARGUMENTS["lr0"],
        "device": TRAIN_ARGUMENTS["device"],
        "workers": TRAIN_ARGUMENTS["workers"],
        "seed": TRAIN_ARGUMENTS["seed"],
        "deterministic": TRAIN_ARGUMENTS["deterministic"],
        "project": str(PROJECT_DIR.resolve()),
        "name": RUN_NAME,
        "training_uses_test_set": False,
        "online_augmentation": {
            "pipeline": "Ultralytics YOLOv8 detection v8_transforms",
            "applied_to": "train only",
            "fliplr_probability": TRAIN_ARGUMENTS["fliplr"],
            "degrees_range": [
                -TRAIN_ARGUMENTS["degrees"],
                TRAIN_ARGUMENTS["degrees"],
            ],
            "translate_fraction": TRAIN_ARGUMENTS["translate"],
            "scale_fraction": TRAIN_ARGUMENTS["scale"],
            "brightness_contrast": BRIGHTNESS_CONTRAST,
            "disabled_detection_augmentations": {
                key: TRAIN_ARGUMENTS[key]
                for key in (
                    "flipud",
                    "shear",
                    "perspective",
                    "hsv_h",
                    "hsv_s",
                    "hsv_v",
                    "mosaic",
                    "mixup",
                    "copy_paste",
                    "cutmix",
                    "bgr",
                    "multi_scale",
                    "erasing",
                )
            },
            "auto_augment": None,
        },
        "train_arguments": TRAIN_ARGUMENTS,
        "test_evaluation": {
            "split": "test",
            "imgsz": 640,
            "batch": 4,
            "device": 0,
            "workers": 2,
            "augment": False,
        },
    }


def write_comparison(
    metrics_by_model: dict[str, dict[str, float] | None],
) -> None:
    baseline = metrics_by_model["M0"]
    if baseline is None:
        raise ValueError("M0 metrics are required for the comparison")
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "Model",
                "Augmentation",
                *(label for _, label in METRIC_COLUMNS),
                *(f"Delta_vs_M0_{label}" for _, label in METRIC_COLUMNS),
            ]
        )
        for name in ("M0", "M1", "M2", "M3", "M4", "M5"):
            metrics = metrics_by_model[name]
            if metrics is None:
                writer.writerow([name, EXPERIMENT_NAMES[name], *[""] * 8])
                continue
            writer.writerow(
                [
                    name,
                    EXPERIMENT_NAMES[name],
                    *(f"{metrics[key]:.9f}" for key, _ in METRIC_COLUMNS),
                    *(
                        f"{metrics[key] - baseline[key]:+.9f}"
                        for key, _ in METRIC_COLUMNS
                    ),
                ]
            )


def pending_test_report() -> dict[str, Any]:
    return {
        "experiment": RUN_NAME,
        "status": "pending_training",
        "split": "test",
        "dataset": str(DATASET_CONFIG.resolve()),
        "test_images": SPLIT_COUNTS["test"],
        "metrics": None,
        "training_seconds": None,
        "summary_vs_M3": None,
        "best_models_M0_M5": None,
    }


def pending_report_text() -> str:
    return (
        "M5 CONTROLLED ONLINE AUGMENTATION — TEST RESULTS\n"
        "Status: pending training (training has not been started)\n"
        f"Dataset: {DATASET_CONFIG.resolve()}\n"
        "Split: test (92 images)\n"
        "Precision: pending\n"
        "Recall: pending\n"
        "mAP@50: pending\n"
        "mAP@50-95: pending\n"
        "Training time: pending\n"
    )


def preflight() -> None:
    validate_worktree()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Pretrained YOLOv8n weights not found: {MODEL_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required; no CUDA device is available")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA device 0 is unavailable")
    validate_dependency_and_api()
    counts = validate_dataset()
    metrics = {name: load_test_metrics(name) for name in ("M0", "M1", "M2", "M3", "M4")}
    pending_exists = check_output_paths(allow_pending=True)

    if not pending_exists:
        config = base_config(counts)
        test_report = pending_test_report()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        write_json(CONFIG_PATH, config)
        write_json(TEST_JSON_PATH, test_report)
        atomic_write_text(TEST_TEXT_PATH, pending_report_text())
        write_comparison({**metrics, "M5": None})

    print("M5_PREFLIGHT: passed")
    print(f"Git root: {ROOT.resolve()}")
    print(f"Branch: {git_output('branch', '--show-current')}")
    print(f"Dataset: {DATASET_CONFIG.resolve()} ({counts})")
    print(f"Model weights: {MODEL_PATH.resolve()}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"Albumentations: {A.__version__}")
    print("Training: not started")
    print(f"Config: {CONFIG_PATH.resolve()}")
    print(f"Pending test report: {TEST_JSON_PATH.resolve()}")
    print(f"Comparison: {COMPARISON_PATH.resolve()}")


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


def comparison_summary(
    all_metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    m3 = all_metrics["M3"]
    m5 = all_metrics["M5"]
    return {
        "M5_beats_M3": {
            "mAP50": m5["mAP50"] > m3["mAP50"],
            "mAP50_95": m5["mAP50_95"] > m3["mAP50_95"],
            "precision": m5["precision"] > m3["precision"],
            "recall": m5["recall"] > m3["recall"],
        },
        "best_by_mAP50_95": max(
            all_metrics, key=lambda name: all_metrics[name]["mAP50_95"]
        ),
        "best_by_mAP50": max(
            all_metrics, key=lambda name: all_metrics[name]["mAP50"]
        ),
    }


def train_and_evaluate() -> None:
    preflight()
    counts = validate_dataset()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    pending = json.loads(TEST_JSON_PATH.read_text(encoding="utf-8"))
    if (
        config.get("preflight_status") != "passed"
        or pending.get("status") != "pending_training"
    ):
        raise RuntimeError("M5 reports are not in a valid preflight-passed state")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite M5 output: {OUTPUT_DIR}")

    config["training_status"] = "running"
    write_json(CONFIG_PATH, config)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    model = YOLO(str(MODEL_PATH))
    training_results = model.train(
        data=str(DATASET_CONFIG),
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=False,
        trainer=M5DetectionTrainer,
        **TRAIN_ARGUMENTS,
    )
    training_seconds = time.perf_counter() - started
    result_dir = Path(training_results.save_dir)
    best_weights = result_dir / "weights" / "best.pt"
    args_path = result_dir / "args.yaml"
    results_path = result_dir / "results.csv"
    missing = [path for path in (best_weights, args_path, results_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required M5 training outputs missing: {missing}")
    with results_path.open(newline="", encoding="utf-8-sig") as stream:
        training_rows = list(csv.DictReader(stream))
    if len(training_rows) != TRAIN_ARGUMENTS["epochs"]:
        raise RuntimeError(
            f"Expected {TRAIN_ARGUMENTS['epochs']} completed epochs, "
            f"found {len(training_rows)}"
        )

    saved_args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    expected_args = {
        key: TRAIN_ARGUMENTS[key]
        for key in (
            "epochs",
            "imgsz",
            "batch",
            "workers",
            "seed",
            "deterministic",
            "pretrained",
            "fliplr",
            "degrees",
            "translate",
            "scale",
            "auto_augment",
            "mosaic",
            "mixup",
        )
    }
    expected_args["device"] = str(TRAIN_ARGUMENTS["device"])
    mismatches = {
        key: {"expected": value, "actual": saved_args.get(key)}
        for key, value in expected_args.items()
        if saved_args.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Saved M5 arguments do not match the plan: {mismatches}")

    test_metrics_result = YOLO(str(best_weights)).val(
        data=str(DATASET_CONFIG),
        split="test",
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project=str(OUTPUT_DIR / "evaluation"),
        name="test",
        exist_ok=False,
        plots=True,
        save_json=False,
        augment=False,
    )
    test_metrics = metric_summary(test_metrics_result)
    historical_metrics = {
        name: load_test_metrics(name) for name in ("M0", "M1", "M2", "M3", "M4")
    }
    all_metrics = {**historical_metrics, "M5": test_metrics}
    summary = comparison_summary(all_metrics)
    config.update(
        {
            "training_status": "completed",
            "training_seconds": training_seconds,
            "training_save_dir": str(result_dir.resolve()),
            "completed_epochs": len(training_rows),
            "training_metrics_last_epoch": training_rows[-1],
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "test_results": test_metrics,
            "summary_vs_M3": summary["M5_beats_M3"],
            "best_models_M0_M5": {
                "mAP50_95": summary["best_by_mAP50_95"],
                "mAP50": summary["best_by_mAP50"],
            },
        }
    )
    test_report = {
        "experiment": RUN_NAME,
        "status": "completed",
        "model": str(best_weights.resolve()),
        "dataset": str(DATASET_CONFIG.resolve()),
        "split": "test",
        "test_images": counts["test"],
        "metrics": test_metrics,
        "training_seconds": training_seconds,
        "completed_epochs": len(training_rows),
        "summary_vs_M3": summary["M5_beats_M3"],
        "best_models_M0_M5": {
            "mAP50_95": summary["best_by_mAP50_95"],
            "mAP50": summary["best_by_mAP50"],
        },
    }
    write_comparison(all_metrics)
    write_json(CONFIG_PATH, config)
    write_json(TEST_JSON_PATH, test_report)
    lines = [
        "M5 CONTROLLED ONLINE AUGMENTATION — FINAL TEST EVALUATION",
        f"Model: {test_report['model']}",
        f"Dataset: {test_report['dataset']}",
        f"Split: test ({counts['test']} images)",
        f"Training time: {training_seconds:.2f} seconds",
        f"Precision: {test_metrics['precision']:.6f}",
        f"Recall: {test_metrics['recall']:.6f}",
        f"mAP@50: {test_metrics['mAP50']:.6f}",
        f"mAP@50-95: {test_metrics['mAP50_95']:.6f}",
        "",
        "M5 compared with M3:",
        *(
            f"{metric}: {status}"
            for metric, status in summary["M5_beats_M3"].items()
        ),
        f"Best M0-M5 by mAP50-95: {summary['best_by_mAP50_95']}",
        f"Best M0-M5 by mAP50: {summary['best_by_mAP50']}",
        "",
        f"Comparison CSV: {COMPARISON_PATH.resolve()}",
    ]
    atomic_write_text(TEST_TEXT_PATH, "\n".join(lines) + "\n")
    print("M5_TRAINING_COMPLETED: true")
    print(f"M5_COMPLETED_EPOCHS: {len(training_rows)}")
    print(f"M5_TRAINING_SECONDS: {training_seconds:.2f}")
    print("M5_TEST_METRICS: " + json.dumps(test_metrics))
    print("M5_COMPARISON_SUMMARY: " + json.dumps(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--train", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.preflight:
        preflight()
    else:
        train_and_evaluate()


if __name__ == "__main__":
    main()
