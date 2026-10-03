#!/usr/bin/env python3
"""Audit the created M1 dataset without modifying any dataset files."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from create_m1_flip_dataset import (
    CLASS_NAMES,
    EPSILON,
    EXPECTED_COUNTS,
    IMAGE_EXTENSIONS,
    OUTPUT_ROOT,
    SOURCE_ROOT,
)


ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
JSON_REPORT = REPORTS_DIR / "M1_dataset_audit.json"
TEXT_REPORT = REPORTS_DIR / "M1_dataset_audit.txt"


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


def audit_label(
    label_path: Path,
    findings: dict[str, list[dict[str, Any]]],
    objects: Counter[int],
) -> None:
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        findings["label_read_errors"].append(
            {"file": str(label_path), "error": str(exc)}
        )
        return
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        findings["empty_labels"].append({"file": str(label_path)})
        return
    for line_number, line in enumerate(nonempty, start=1):
        fields = line.split()
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
            objects[class_id] += 1
        for name, value, valid in (
            ("x_center", x_center, math.isfinite(x_center) and 0.0 <= x_center <= 1.0),
            ("y_center", y_center, math.isfinite(y_center) and 0.0 <= y_center <= 1.0),
            ("width", width, math.isfinite(width) and 0.0 < width <= 1.0),
            ("height", height, math.isfinite(height) and 0.0 < height <= 1.0),
        ):
            if not valid:
                findings[f"{name}_range_errors"].append(
                    {"file": location, "value": value}
                )
        if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
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
                    "bbox_normalized": {
                        "left": left,
                        "right": right,
                        "top": top,
                        "bottom": bottom,
                    },
                }
            )


def audit_dataset() -> dict[str, Any]:
    if not OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(f"M1 dataset not found: {OUTPUT_ROOT}")
    expected_splits = ("train", "val", "test")
    findings: dict[str, list[dict[str, Any]]] = {
        key: []
        for key in (
            "image_label_mismatches",
            "empty_labels",
            "label_read_errors",
            "label_parse_errors",
            "invalid_class_ids",
            "x_center_range_errors",
            "y_center_range_errors",
            "width_range_errors",
            "height_range_errors",
            "bbox_containment_errors",
            "corrupt_images",
            "duplicate_image_hashes",
            "train_val_hash_leakage",
            "train_test_hash_leakage",
            "validation_source_mismatches",
            "test_source_mismatches",
            "dataset_yaml_errors",
        )
    }
    split_results: dict[str, Any] = {}
    image_hash_locations: dict[str, list[str]] = defaultdict(list)
    split_hashes: dict[str, dict[str, str]] = {}
    class_counts: Counter[int] = Counter()
    all_resolutions: Counter[str] = Counter()

    for split in expected_splits:
        split_class_counts: Counter[int] = Counter()
        split_root = OUTPUT_ROOT / split
        images_dir = split_root / "images"
        labels_dir = split_root / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Incomplete M1 split: {split_root}")
        images = sorted(
            (
                path
                for path in images_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.lower(),
        )
        labels = sorted(
            (path for path in labels_dir.glob("*.txt") if path.is_file()),
            key=lambda path: path.name.lower(),
        )
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            findings["image_label_mismatches"].append(
                {
                    "split": split,
                    "images_without_labels": sorted(image_stems - label_stems),
                    "labels_without_images": sorted(label_stems - image_stems),
                }
            )

        for label in labels:
            audit_label(label, findings, split_class_counts)
        class_counts.update(split_class_counts)

        resolutions: Counter[str] = Counter()
        hashes: dict[str, str] = {}
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    resolution = f"{image.width}x{image.height}"
                    image.verify()
                resolutions[resolution] += 1
                all_resolutions[resolution] += 1
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                findings["corrupt_images"].append(
                    {"file": str(image_path), "error": str(exc)}
                )
                continue
            digest = sha256_file(image_path)
            hashes[image_path.stem] = digest
            image_hash_locations[digest].append(f"{split}/{image_path.name}")
        split_hashes[split] = hashes
        split_results[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": {
                "helmet": split_class_counts[0],
                "no-helmet": split_class_counts[1],
            },
            "resolutions": dict(sorted(resolutions.items())),
        }

    duplicates = [
        {"sha256": digest, "images": locations}
        for digest, locations in image_hash_locations.items()
        if len(locations) > 1
    ]
    findings["duplicate_image_hashes"].extend(duplicates)
    for train_stem, train_hash in split_hashes["train"].items():
        for other_split, finding_key in (
            ("val", "train_val_hash_leakage"),
            ("test", "train_test_hash_leakage"),
        ):
            for other_stem, other_hash in split_hashes[other_split].items():
                if train_hash == other_hash:
                    findings[finding_key].append(
                        {
                            "train_image": f"train/{train_stem}",
                            f"{other_split}_image": f"{other_split}/{other_stem}",
                            "sha256": train_hash,
                        }
                    )

    for split in ("val", "test"):
        source_split = SOURCE_ROOT / split
        target_split = OUTPUT_ROOT / split
        source_files = {
            path.relative_to(source_split).as_posix(): path
            for path in source_split.rglob("*")
            if path.is_file()
            and (
                path.parent.name in ("images", "labels")
                and (
                    path.suffix.lower() in IMAGE_EXTENSIONS
                    or path.suffix.lower() == ".txt"
                )
            )
        }
        for relative, source_path in source_files.items():
            target_path = target_split.joinpath(*Path(relative).parts)
            if (
                not target_path.is_file()
                or sha256_file(source_path) != sha256_file(target_path)
            ):
                findings[f"{split}_source_mismatches"].append(
                    {"source": str(source_path), "M1": str(target_path)}
                )

    data_yaml_path = OUTPUT_ROOT / "data.yaml"
    try:
        data_config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
        names = data_config.get("names")
        if isinstance(names, list):
            names = {index: name for index, name in enumerate(names)}
        if (
            data_config.get("train") != "train/images"
            or data_config.get("val") != "val/images"
            or data_config.get("test") != "test/images"
            or data_config.get("nc") != 2
            or names != CLASS_NAMES
        ):
            findings["dataset_yaml_errors"].append(
                {"file": str(data_yaml_path), "parsed_config": data_config}
            )
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        findings["dataset_yaml_errors"].append(
            {"file": str(data_yaml_path), "error": str(exc)}
        )

    expected_counts = {"train": 635, "val": 181, "test": 92}
    for split, expected in expected_counts.items():
        if split_results[split]["images"] != expected or split_results[split]["labels"] != expected:
            findings["image_label_mismatches"].append(
                {
                    "split": split,
                    "expected_images_and_labels": expected,
                    "actual_images": split_results[split]["images"],
                    "actual_labels": split_results[split]["labels"],
                }
            )
    error_count = sum(len(entries) for entries in findings.values())
    report = {
        "experiment": "M1_flip",
        "dataset": str(OUTPUT_ROOT.resolve()),
        "source_dataset": str(SOURCE_ROOT.resolve()),
        "epsilon_normalized": EPSILON,
        "split_counts": split_results,
        "class_object_counts_total": {
            "helmet": sum(
                split_results[split]["objects"]["helmet"] for split in expected_splits
            ),
            "no-helmet": sum(
                split_results[split]["objects"]["no-helmet"]
                for split in expected_splits
            ),
        },
        "resolutions_all_splits": dict(sorted(all_resolutions.items())),
        "findings": findings,
        "error_count": error_count,
        "passed": error_count == 0,
    }
    return report


def write_text_report(report: dict[str, Any]) -> None:
    lines = [
        "M1 DATASET AUDIT",
        f"Dataset: {report['dataset']}",
        f"Source dataset: {report['source_dataset']}",
        f"Normalized bbox containment epsilon: {report['epsilon_normalized']}",
        "",
    ]
    for split in ("train", "val", "test"):
        details = report["split_counts"][split]
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
            f"  Annotation / label errors: {sum(len(report['findings'][key]) for key in ('empty_labels', 'label_read_errors', 'label_parse_errors', 'invalid_class_ids', 'x_center_range_errors', 'y_center_range_errors', 'width_range_errors', 'height_range_errors', 'bbox_containment_errors'))}",
            f"  Corrupt images: {len(report['findings']['corrupt_images'])}",
            f"  Duplicate image SHA-256 groups: {len(report['findings']['duplicate_image_hashes'])}",
            f"  Image-label/count mismatches: {len(report['findings']['image_label_mismatches'])}",
            f"  Validation/test source mismatches: {len(report['findings']['validation_source_mismatches']) + len(report['findings']['test_source_mismatches'])}",
            f"  Train-vs-validation duplicate hashes: {len(report['findings']['train_val_hash_leakage'])}",
            f"  Train-vs-test duplicate hashes: {len(report['findings']['train_test_hash_leakage'])}",
            f"  Dataset YAML errors: {len(report['findings']['dataset_yaml_errors'])}",
            f"TOTAL ERRORS: {report['error_count']}",
            f"PASSED: {str(report['passed']).lower()}",
        ]
    )
    if report["passed"]:
        lines.append("M1 dataset READY FOR TRAINING")
    else:
        lines.append("")
        lines.append("FINDINGS:")
        lines.append(json.dumps(report["findings"], ensure_ascii=False, indent=2))
    TEXT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = audit_dataset()
    write_json(JSON_REPORT, report)
    write_text_report(report)
    print(f"M1_AUDIT_JSON: {JSON_REPORT.resolve()}")
    print(f"M1_AUDIT_TEXT: {TEXT_REPORT.resolve()}")
    print(f"M1_AUDIT_PASSED: {str(report['passed']).lower()}")
    print(f"M1_AUDIT_ERROR_COUNT: {report['error_count']}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
