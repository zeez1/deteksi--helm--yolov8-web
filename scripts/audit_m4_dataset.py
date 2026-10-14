#!/usr/bin/env python3
"""Audit only the generated M4 dataset and write its audit reports."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "dataset" / "experiments" / "M4_compound"
REPORTS_DIR = ROOT / "reports"
JSON_REPORT = REPORTS_DIR / "M4_dataset_audit.json"
TEXT_REPORT = REPORTS_DIR / "M4_dataset_audit.txt"
EXPECTED_COUNTS = {"train": 635, "val": 181, "test": 92}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_NAMES = {0: "helmet", 1: "no-helmet"}
EPSILON = 1e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def empty_findings() -> dict[str, list[dict[str, Any]]]:
    return {
        name: []
        for name in (
            "structure_errors",
            "image_label_mismatches",
            "empty_labels",
            "label_read_errors",
            "label_parse_errors",
            "invalid_class_ids",
            "bbox_validity_errors",
            "bbox_containment_errors",
            "corrupt_images",
            "duplicate_image_sha256",
            "train_val_leakage",
            "train_test_leakage",
            "val_test_leakage",
            "dataset_yaml_errors",
            "metadata_errors",
        )
    }


def audit_label(
    label_path: Path,
    findings: dict[str, list[dict[str, Any]]],
    object_counts: Counter[int],
) -> None:
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        findings["label_read_errors"].append(
            {"file": str(label_path), "error": str(exc)}
        )
        return
    rows = [line for line in lines if line.strip()]
    if not rows:
        findings["empty_labels"].append({"file": str(label_path)})
        return

    for line_number, line in enumerate(rows, start=1):
        location = f"{label_path}:{line_number}"
        fields = line.split()
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
            object_counts[class_id] += 1

        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in coordinates) or not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            findings["bbox_validity_errors"].append(
                {"file": location, "values": list(coordinates)}
            )
            if not all(math.isfinite(value) for value in coordinates):
                continue
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


def audit_dataset() -> dict[str, Any]:
    findings = empty_findings()
    expected_root_entries = {
        "train",
        "val",
        "test",
        "data.yaml",
        "dataset_info.json",
    }
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"M4 dataset not found: {DATASET_ROOT}")
    actual_root_entries = {path.name for path in DATASET_ROOT.iterdir()}
    if actual_root_entries != expected_root_entries:
        findings["structure_errors"].append(
            {
                "expected_root_entries": sorted(expected_root_entries),
                "actual_root_entries": sorted(actual_root_entries),
            }
        )

    split_counts: dict[str, Any] = {}
    split_hashes: dict[str, dict[str, str]] = {}
    hash_locations: dict[str, list[str]] = defaultdict(list)
    total_objects: Counter[int] = Counter()
    all_resolutions: Counter[str] = Counter()
    for split, expected_count in EXPECTED_COUNTS.items():
        split_root = DATASET_ROOT / split
        images_dir = split_root / "images"
        labels_dir = split_root / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            findings["structure_errors"].append(
                {"split": split, "error": "missing images or labels directory"}
            )
            split_counts[split] = {
                "images": 0,
                "labels": 0,
                "objects": {"helmet": 0, "no-helmet": 0},
                "resolutions": {},
            }
            split_hashes[split] = {}
            continue
        split_entries = {path.name for path in split_root.iterdir()}
        if split_entries != {"images", "labels"}:
            findings["structure_errors"].append(
                {
                    "split": split,
                    "expected_entries": ["images", "labels"],
                    "actual_entries": sorted(split_entries),
                }
            )

        images = image_files(images_dir)
        labels = sorted(
            (path for path in labels_dir.glob("*.txt") if path.is_file()),
            key=lambda path: path.name.lower(),
        )
        image_stems = [path.stem for path in images]
        label_stems = [path.stem for path in labels]
        if (
            len(images) != expected_count
            or len(labels) != expected_count
            or len(set(image_stems)) != len(image_stems)
            or len(set(label_stems)) != len(label_stems)
            or set(image_stems) != set(label_stems)
        ):
            findings["image_label_mismatches"].append(
                {
                    "split": split,
                    "expected_images_and_labels": expected_count,
                    "images": len(images),
                    "labels": len(labels),
                    "images_without_labels": sorted(set(image_stems) - set(label_stems)),
                    "labels_without_images": sorted(set(label_stems) - set(image_stems)),
                }
            )

        split_objects: Counter[int] = Counter()
        for label in labels:
            audit_label(label, findings, split_objects)
        total_objects.update(split_objects)

        resolutions: Counter[str] = Counter()
        hashes: dict[str, str] = {}
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    resolution = f"{image.width}x{image.height}"
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                findings["corrupt_images"].append(
                    {"file": str(image_path), "error": str(exc)}
                )
                continue
            resolutions[resolution] += 1
            all_resolutions[resolution] += 1
            digest = sha256_file(image_path)
            hashes[image_path.stem] = digest
            hash_locations[digest].append(f"{split}/{image_path.name}")
        split_hashes[split] = hashes
        split_counts[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": {
                "helmet": split_objects[0],
                "no-helmet": split_objects[1],
            },
            "resolutions": dict(sorted(resolutions.items())),
        }

    findings["duplicate_image_sha256"] = [
        {"sha256": digest, "images": locations}
        for digest, locations in hash_locations.items()
        if len(locations) > 1
    ]
    for first, second, finding_name in (
        ("train", "val", "train_val_leakage"),
        ("train", "test", "train_test_leakage"),
        ("val", "test", "val_test_leakage"),
    ):
        first_hashes = set(split_hashes.get(first, {}).values())
        second_hashes = set(split_hashes.get(second, {}).values())
        overlap = first_hashes & second_hashes
        if overlap:
            findings[finding_name].extend(
                {
                    "sha256": digest,
                    "first_split_images": [
                        f"{first}/{stem}"
                        for stem, value in split_hashes[first].items()
                        if value == digest
                    ],
                    "second_split_images": [
                        f"{second}/{stem}"
                        for stem, value in split_hashes[second].items()
                        if value == digest
                    ],
                }
                for digest in sorted(overlap)
            )

    config_path = DATASET_ROOT / "data.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        names = config.get("names")
        if isinstance(names, list):
            names = dict(enumerate(names))
        configured_root = Path(config.get("path", ""))
        if not configured_root.is_absolute():
            configured_root = (DATASET_ROOT / configured_root).resolve()
        if (
            not isinstance(config, dict)
            or configured_root.resolve() != DATASET_ROOT.resolve()
            or config.get("train") != "train/images"
            or config.get("val") != "val/images"
            or config.get("test") != "test/images"
            or config.get("nc") != len(CLASS_NAMES)
            or names != CLASS_NAMES
        ):
            findings["dataset_yaml_errors"].append(
                {"file": str(config_path), "parsed_config": config}
            )
    except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError) as exc:
        findings["dataset_yaml_errors"].append(
            {"file": str(config_path), "error": str(exc)}
        )

    metadata_path = DATASET_ROOT / "dataset_info.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata_counts = {
            split: {
                "images": split_counts[split]["images"],
                "labels": split_counts[split]["labels"],
                "objects": split_counts[split]["objects"],
            }
            for split in EXPECTED_COUNTS
        }
        if (
            metadata.get("experiment") != "M4_compound"
            or metadata.get("split_counts") != expected_metadata_counts
            or metadata.get("augmentation", {}).get("applied_to")
            != "train only; one compound-transformed image per source image"
            or metadata.get("validation_test_copied_unchanged") is not True
        ):
            findings["metadata_errors"].append(
                {"file": str(metadata_path), "error": "metadata differs from audited data"}
            )
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError, TypeError) as exc:
        findings["metadata_errors"].append(
            {"file": str(metadata_path), "error": str(exc)}
        )

    error_count = sum(len(entries) for entries in findings.values())
    return {
        "experiment": "M4_compound",
        "dataset": str(DATASET_ROOT.resolve()),
        "augmentation": {
            "sequence": [
                "horizontal_flip",
                "rotation",
                "brightness",
                "contrast",
            ],
            "rotation_degrees_range": [-15.0, 15.0],
            "brightness_factor": 0.8,
            "contrast_factor": 1.2,
            "applied_to": "train only",
        },
        "epsilon_normalized": EPSILON,
        "split_counts": split_counts,
        "class_object_counts_total": {
            "helmet": total_objects[0],
            "no-helmet": total_objects[1],
        },
        "resolutions_all_splits": dict(sorted(all_resolutions.items())),
        "findings": findings,
        "error_count": error_count,
        "passed": error_count == 0,
    }


def write_reports(report: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "M4 COMPOUND DATASET AUDIT",
        f"Dataset: {report['dataset']}",
        "Augmentation: horizontal flip + rotation [-15, +15] degrees "
        "+ brightness 0.8 + contrast 1.2",
        "Applied to: train only",
        f"Normalized bbox containment epsilon: {EPSILON}",
        "",
    ]
    for split, details in report["split_counts"].items():
        lines.extend(
            [
                f"{split.upper()}:",
                f"  Images: {details['images']}",
                f"  Labels: {details['labels']}",
                f"  Helmet objects: {details['objects']['helmet']}",
                f"  No-helmet objects: {details['objects']['no-helmet']}",
                f"  Resolutions: {json.dumps(details['resolutions'], sort_keys=True)}",
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
    TEXT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = audit_dataset()
    write_reports(report)
    print(f"M4_AUDIT_PASSED: {str(report['passed']).lower()}")
    print(f"M4_AUDIT_JSON: {JSON_REPORT.resolve()}")
    print(f"M4_AUDIT_TEXT: {TEXT_REPORT.resolve()}")
    print(f"M4_AUDIT_ERRORS: {report['error_count']}")
    if not report["passed"]:
        raise RuntimeError(f"M4 dataset audit failed: {JSON_REPORT}")


if __name__ == "__main__":
    main()
