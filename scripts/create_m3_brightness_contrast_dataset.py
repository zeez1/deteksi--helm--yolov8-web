#!/usr/bin/env python3
"""Create and audit the M3 brightness-and-contrast dataset."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageEnhance, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "dataset" / "helmet_research"
OUTPUT_ROOT = ROOT / "dataset" / "experiments" / "M3_brightness_contrast"
STAGING_ROOT = OUTPUT_ROOT.with_name(".M3_brightness_contrast.staging")
REPORTS_DIR = ROOT / "reports"
AUDIT_JSON = REPORTS_DIR / "M3_dataset_audit.json"
AUDIT_TEXT = REPORTS_DIR / "M3_dataset_audit.txt"
SEED = 42
BRIGHTNESS_FACTOR = 0.8
CONTRAST_FACTOR = 1.2
EPSILON = 1e-5
EXPECTED_COUNTS = {"train": 635, "val": 181, "test": 92}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "helmet", 1: "no-helmet"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def audit_label(
    label_path: Path,
    findings: dict[str, list[dict[str, Any]]],
    class_counts: Counter[int],
) -> None:
    try:
        rows = [
            line.split()
            for line in label_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as exc:
        findings["label_read_errors"].append(
            {"file": str(label_path), "error": str(exc)}
        )
        return
    if not rows:
        findings["empty_labels"].append({"file": str(label_path)})
        return

    for line_number, fields in enumerate(rows, start=1):
        location = f"{label_path}:{line_number}"
        if len(fields) != 5:
            findings["label_parse_errors"].append(
                {"file": location, "error": f"expected 5 fields, found {len(fields)}"}
            )
            continue
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError as exc:
            findings["label_parse_errors"].append(
                {"file": location, "error": str(exc)}
            )
            continue

        if class_id not in CLASS_NAMES:
            findings["invalid_class_ids"].append(
                {"file": location, "class_id": class_id}
            )
        else:
            class_counts[class_id] += 1

        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in coordinates):
            findings["bbox_normalized_validity_errors"].append(
                {"file": location, "values": list(coordinates)}
            )
            continue
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            findings["bbox_normalized_validity_errors"].append(
                {"file": location, "values": list(coordinates)}
            )
        left = x_center - width / 2
        right = x_center + width / 2
        top = y_center - height / 2
        bottom = y_center + height / 2
        if (
            left < -EPSILON
            or right > 1.0 + EPSILON
            or top < -EPSILON
            or bottom > 1.0 + EPSILON
        ):
            findings["bbox_containment_errors"].append(
                {
                    "file": location,
                    "bbox_normalized": [left, top, right, bottom],
                }
            )


def audit_tree(root: Path, *, compare_validation_test_to_source: bool) -> dict[str, Any]:
    finding_names = (
        "image_label_mismatches",
        "empty_labels",
        "label_read_errors",
        "label_parse_errors",
        "invalid_class_ids",
        "bbox_normalized_validity_errors",
        "bbox_containment_errors",
        "corrupt_images",
        "duplicate_image_sha256",
        "train_validation_leakage",
        "train_test_leakage",
        "validation_source_mismatches",
        "test_source_mismatches",
        "dataset_yaml_errors",
    )
    findings: dict[str, list[dict[str, Any]]] = {
        name: [] for name in finding_names
    }
    split_results: dict[str, Any] = {}
    split_hashes: dict[str, dict[str, str]] = {}
    hash_locations: dict[str, list[str]] = defaultdict(list)
    total_class_counts: Counter[int] = Counter()
    resolutions: Counter[str] = Counter()

    for split, expected in EXPECTED_COUNTS.items():
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Incomplete {split} split: {root / split}")
        images = image_files(images_dir)
        labels = sorted(
            (path for path in labels_dir.glob("*.txt") if path.is_file()),
            key=lambda path: path.name.lower(),
        )
        image_stems = [path.stem for path in images]
        label_stems = [path.stem for path in labels]
        if (
            len(images) != expected
            or len(labels) != expected
            or set(image_stems) != set(label_stems)
            or len(set(image_stems)) != len(image_stems)
            or len(set(label_stems)) != len(label_stems)
        ):
            findings["image_label_mismatches"].append(
                {
                    "split": split,
                    "expected_images_and_labels": expected,
                    "actual_images": len(images),
                    "actual_labels": len(labels),
                    "images_without_labels": sorted(set(image_stems) - set(label_stems)),
                    "labels_without_images": sorted(set(label_stems) - set(image_stems)),
                }
            )

        class_counts: Counter[int] = Counter()
        for label in labels:
            audit_label(label, findings, class_counts)
        total_class_counts.update(class_counts)

        split_hashes[split] = {}
        split_resolutions: Counter[str] = Counter()
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    size = f"{image.width}x{image.height}"
                    image.verify()
                split_resolutions[size] += 1
                resolutions[size] += 1
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                findings["corrupt_images"].append(
                    {"file": str(image_path), "error": str(exc)}
                )
                continue
            digest = sha256_file(image_path)
            split_hashes[split][image_path.stem] = digest
            hash_locations[digest].append(f"{split}/{image_path.name}")
        split_results[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": {
                "helmet": class_counts[0],
                "no-helmet": class_counts[1],
            },
            "resolutions": dict(sorted(split_resolutions.items())),
        }

    findings["duplicate_image_sha256"] = [
        {"sha256": digest, "images": locations}
        for digest, locations in hash_locations.items()
        if len(locations) > 1
    ]
    for split, finding_name in (
        ("val", "train_validation_leakage"),
        ("test", "train_test_leakage"),
    ):
        overlap = set(split_hashes["train"].values()) & set(
            split_hashes[split].values()
        )
        for digest in overlap:
            findings[finding_name].append(
                {
                    "sha256": digest,
                    "train_images": [
                        name
                        for name, value in split_hashes["train"].items()
                        if value == digest
                    ],
                    f"{split}_images": [
                        name
                        for name, value in split_hashes[split].items()
                        if value == digest
                    ],
                }
            )

    if compare_validation_test_to_source:
        for split in ("val", "test"):
            source_split = SOURCE_ROOT / split
            output_split = root / split
            for source_file in source_split.rglob("*"):
                if (
                    not source_file.is_file()
                    or source_file.parent.name not in ("images", "labels")
                    or not (
                        source_file.suffix.lower() in IMAGE_EXTENSIONS
                        or source_file.suffix.lower() == ".txt"
                    )
                ):
                    continue
                output_file = output_split / source_file.relative_to(source_split)
                if (
                    not output_file.is_file()
                    or sha256_file(source_file) != sha256_file(output_file)
                ):
                    findings[f"{split}_source_mismatches"].append(
                        {"source": str(source_file), "target": str(output_file)}
                    )

    try:
        dataset_config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
        names = dataset_config.get("names")
        if isinstance(names, list):
            names = dict(enumerate(names))
        if (
            dataset_config.get("train") != "train/images"
            or dataset_config.get("val") != "val/images"
            or dataset_config.get("test") != "test/images"
            or dataset_config.get("nc") != 2
            or names != CLASS_NAMES
        ):
            findings["dataset_yaml_errors"].append(
                {"file": str(root / "data.yaml"), "parsed_config": dataset_config}
            )
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        findings["dataset_yaml_errors"].append(
            {"file": str(root / "data.yaml"), "error": str(exc)}
        )

    error_count = sum(len(entries) for entries in findings.values())
    return {
        "experiment": "M3_brightness_contrast",
        "dataset": str(root.resolve()),
        "source_dataset": str(SOURCE_ROOT.resolve()),
        "brightness_factor": BRIGHTNESS_FACTOR,
        "contrast_factor": CONTRAST_FACTOR,
        "seed": SEED,
        "epsilon_normalized": EPSILON,
        "split_counts": split_results,
        "class_object_counts_total": {
            "helmet": total_class_counts[0],
            "no-helmet": total_class_counts[1],
        },
        "resolutions_all_splits": dict(sorted(resolutions.items())),
        "findings": findings,
        "error_count": error_count,
        "passed": error_count == 0,
    }


def write_audit(report: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "M3 BRIGHTNESS + CONTRAST DATASET AUDIT",
        f"Dataset: {report['dataset']}",
        f"Source dataset: {report['source_dataset']}",
        f"Brightness factor: {BRIGHTNESS_FACTOR}",
        f"Contrast factor: {CONTRAST_FACTOR}",
        f"Seed: {SEED}",
        f"Normalized bbox containment epsilon: {EPSILON}",
        "",
    ]
    for split, details in report.get("split_counts", {}).items():
        lines.extend(
            [
                f"{split.upper()}:",
                f"  Images: {details['images']}",
                f"  Labels: {details['labels']}",
                f"  Helmet objects: {details['objects']['helmet']}",
                f"  No-helmet objects: {details['objects']['no-helmet']}",
            ]
        )
    lines.extend(
        [
            "",
            "CHECKS:",
            *(f"  {name}: {len(entries)}" for name, entries in report["findings"].items()),
            f"TOTAL ERRORS: {report['error_count']}",
            f"PASSED: {str(report['passed']).lower()}",
        ]
    )
    if not report["passed"]:
        lines.extend(
            ["", "FINDINGS:", json.dumps(report["findings"], ensure_ascii=False, indent=2)]
        )
    AUDIT_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_source() -> dict[str, Any]:
    config_path = SOURCE_ROOT / "data.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Source dataset YAML not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names != CLASS_NAMES:
        raise ValueError(f"Unexpected source class names: {names!r}")
    return audit_tree(SOURCE_ROOT, compare_validation_test_to_source=False)


def create_dataset() -> None:
    random.seed(SEED)
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite M3 dataset: {OUTPUT_ROOT}")
    if STAGING_ROOT.exists():
        raise FileExistsError(f"Staging directory already exists: {STAGING_ROOT}")

    source_report = validate_source()
    if not source_report["passed"]:
        write_audit(source_report)
        raise RuntimeError(
            f"Source dataset audit failed; stopping without modification: {AUDIT_JSON}"
        )

    source_train = image_files(SOURCE_ROOT / "train" / "images")
    STAGING_ROOT.mkdir(parents=True)
    try:
        for split in EXPECTED_COUNTS:
            (STAGING_ROOT / split / "images").mkdir(parents=True)
            (STAGING_ROOT / split / "labels").mkdir(parents=True)

        for source_image in source_train:
            target_image = STAGING_ROOT / "train" / "images" / source_image.name
            with Image.open(source_image) as image:
                brightened = ImageEnhance.Brightness(image).enhance(BRIGHTNESS_FACTOR)
                adjusted = ImageEnhance.Contrast(brightened).enhance(CONTRAST_FACTOR)
                adjusted.save(
                    target_image,
                    format=image.format or target_image.suffix.lstrip(".").upper(),
                )
            source_label = SOURCE_ROOT / "train" / "labels" / f"{source_image.stem}.txt"
            shutil.copy2(
                source_label,
                STAGING_ROOT / "train" / "labels" / source_label.name,
            )

        for split in ("val", "test"):
            for source_file in (SOURCE_ROOT / split).rglob("*"):
                if source_file.is_file() and source_file.parent.name in (
                    "images",
                    "labels",
                ):
                    target_file = STAGING_ROOT / split / source_file.relative_to(
                        SOURCE_ROOT / split
                    )
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, target_file)

        dataset_config = {
            "path": str(OUTPUT_ROOT.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }
        (STAGING_ROOT / "data.yaml").write_text(
            yaml.safe_dump(dataset_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        info = {
            "experiment": "M3_brightness_contrast",
            "source_dataset": str(SOURCE_ROOT.resolve()),
            "augmentation": "brightness + contrast",
            "brightness_factor": BRIGHTNESS_FACTOR,
            "contrast_factor": CONTRAST_FACTOR,
            "seed": SEED,
            "train": 635,
            "validation": 181,
            "test": 92,
            "train_images": 635,
            "train_labels": 635,
            "original_train_images_included": False,
            "horizontal_flip": False,
            "rotation": False,
            "scale": False,
            "other_augmentation": False,
            "brightness_contrast_applied_to": "train only",
            "validation_test_copied_unchanged": True,
            "bounding_boxes_transformed": False,
            "epsilon_normalized": EPSILON,
        }
        (STAGING_ROOT / "dataset_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        STAGING_ROOT.rename(OUTPUT_ROOT)
    except Exception:
        if STAGING_ROOT.exists():
            shutil.rmtree(STAGING_ROOT)
        raise

    report = audit_tree(OUTPUT_ROOT, compare_validation_test_to_source=True)
    write_audit(report)
    print(f"M3_AUDIT_JSON: {AUDIT_JSON.resolve()}")
    print(f"M3_AUDIT_TEXT: {AUDIT_TEXT.resolve()}")
    print(f"M3_DATASET: {OUTPUT_ROOT.resolve()}")
    print(
        "SPLIT_COUNTS: "
        + json.dumps(
            {
                split: (
                    details["images"],
                    details["labels"],
                )
                for split, details in report["split_counts"].items()
            }
        )
    )
    print(f"AUDIT_PASSED: {str(report['passed']).lower()}")
    if not report["passed"]:
        raise RuntimeError(
            f"M3 dataset audit failed; training must not start: {AUDIT_JSON}"
        )


if __name__ == "__main__":
    create_dataset()
