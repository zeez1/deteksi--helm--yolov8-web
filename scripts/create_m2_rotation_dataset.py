#!/usr/bin/env python3
"""Create and audit the M2 dataset with one seeded rotation per train image."""

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
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "dataset" / "helmet_research"
OUTPUT_ROOT = ROOT / "dataset" / "experiments" / "M2_rotation"
STAGING_ROOT = OUTPUT_ROOT.with_name(".M2_rotation.staging")
REPORTS_DIR = ROOT / "reports"
AUDIT_JSON = REPORTS_DIR / "M2_dataset_audit.json"
AUDIT_TEXT = REPORTS_DIR / "M2_dataset_audit.txt"
SEED = 42
DEGREES = 15.0
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
    split: str,
    findings: dict[str, list[dict[str, Any]]],
    counts: Counter[int],
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
            counts[class_id] += 1

        values = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in values):
            findings["bbox_validity_errors"].append(
                {"file": location, "values": list(values)}
            )
            continue
        if not (
            0.0 <= x_center <= 1.0
            and 0.0 <= y_center <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            findings["bbox_validity_errors"].append(
                {"file": location, "values": list(values)}
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


def audit_tree(
    root: Path,
    *,
    verify_val_test_source: bool,
) -> dict[str, Any]:
    keys = (
        "image_label_mismatches",
        "empty_labels",
        "label_read_errors",
        "label_parse_errors",
        "invalid_class_ids",
        "bbox_validity_errors",
        "bbox_containment_errors",
        "corrupt_images",
        "duplicate_image_hashes",
        "train_val_hash_leakage",
        "train_test_hash_leakage",
        "val_source_mismatches",
        "test_source_mismatches",
        "dataset_yaml_errors",
    )
    findings: dict[str, list[dict[str, Any]]] = {key: [] for key in keys}
    split_results: dict[str, Any] = {}
    hashes_by_split: dict[str, dict[str, str]] = {}
    hash_locations: dict[str, list[str]] = defaultdict(list)
    all_class_counts: Counter[int] = Counter()
    all_resolutions: Counter[str] = Counter()

    for split, expected_count in EXPECTED_COUNTS.items():
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
            len(images) != expected_count
            or len(labels) != expected_count
            or set(image_stems) != set(label_stems)
            or len(set(image_stems)) != len(image_stems)
            or len(set(label_stems)) != len(label_stems)
        ):
            findings["image_label_mismatches"].append(
                {
                    "split": split,
                    "expected_each": expected_count,
                    "images": len(images),
                    "labels": len(labels),
                    "images_without_labels": sorted(set(image_stems) - set(label_stems)),
                    "labels_without_images": sorted(set(label_stems) - set(image_stems)),
                }
            )

        split_class_counts: Counter[int] = Counter()
        for label in labels:
            audit_label(label, split, findings, split_class_counts)
        all_class_counts.update(split_class_counts)

        hashes: dict[str, str] = {}
        resolutions: Counter[str] = Counter()
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
            hash_locations[digest].append(f"{split}/{image_path.name}")
        hashes_by_split[split] = hashes
        split_results[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": {
                "helmet": split_class_counts[0],
                "no-helmet": split_class_counts[1],
            },
            "resolutions": dict(sorted(resolutions.items())),
        }

    findings["duplicate_image_hashes"] = [
        {"sha256": digest, "images": locations}
        for digest, locations in hash_locations.items()
        if len(locations) > 1
    ]
    for split, finding_key in (
        ("val", "train_val_hash_leakage"),
        ("test", "train_test_hash_leakage"),
    ):
        common_hashes = set(hashes_by_split["train"].values()) & set(
            hashes_by_split[split].values()
        )
        for digest in common_hashes:
            findings[finding_key].append(
                {
                    "sha256": digest,
                    "train_images": [
                        stem
                        for stem, value in hashes_by_split["train"].items()
                        if value == digest
                    ],
                    f"{split}_images": [
                        stem
                        for stem, value in hashes_by_split[split].items()
                        if value == digest
                    ],
                }
            )

    if verify_val_test_source:
        for split in ("val", "test"):
            source_split = SOURCE_ROOT / split
            target_split = root / split
            for source_file in sorted(
                (
                    path
                    for path in source_split.rglob("*")
                    if path.is_file()
                    and path.parent.name in ("images", "labels")
                    and (
                        path.suffix.lower() in IMAGE_EXTENSIONS
                        or path.suffix.lower() == ".txt"
                    )
                ),
                key=lambda path: path.as_posix().lower(),
            ):
                target = target_split / source_file.relative_to(source_split)
                if not target.is_file() or sha256_file(source_file) != sha256_file(
                    target
                ):
                    findings[f"{split}_source_mismatches"].append(
                        {"source": str(source_file), "target": str(target)}
                    )

    try:
        config = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
        names = config.get("names")
        if isinstance(names, list):
            names = dict(enumerate(names))
        if (
            config.get("train") != "train/images"
            or config.get("val") != "val/images"
            or config.get("test") != "test/images"
            or config.get("nc") != 2
            or names != CLASS_NAMES
        ):
            findings["dataset_yaml_errors"].append(
                {"file": str(root / "data.yaml"), "parsed_config": config}
            )
    except (OSError, yaml.YAMLError, AttributeError) as exc:
        findings["dataset_yaml_errors"].append(
            {"file": str(root / "data.yaml"), "error": str(exc)}
        )

    error_count = sum(len(items) for items in findings.values())
    return {
        "experiment": "M2_rotation",
        "dataset": str(root.resolve()),
        "source_dataset": str(SOURCE_ROOT.resolve()),
        "seed": SEED,
        "rotation_degrees": DEGREES,
        "epsilon_normalized": EPSILON,
        "split_counts": split_results,
        "class_object_counts_total": {
            "helmet": all_class_counts[0],
            "no-helmet": all_class_counts[1],
        },
        "resolutions_all_splits": dict(sorted(all_resolutions.items())),
        "findings": findings,
        "error_count": error_count,
        "passed": error_count == 0,
    }


def source_finding_report() -> dict[str, Any]:
    source_results: dict[str, Any] = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        images = image_files(SOURCE_ROOT / split / "images")
        labels_dir = SOURCE_ROOT / split / "labels"
        labels = sorted(labels_dir.glob("*.txt"), key=lambda path: path.name.lower())
        if len(images) != expected_count or len(labels) != expected_count:
            source_results[split] = {
                "expected_images_and_labels": expected_count,
                "images": len(images),
                "labels": len(labels),
            }
    return source_results


def rotate_box(
    fields: list[str], width: int, height: int, angle_degrees: float
) -> list[str] | None:
    class_id = fields[0]
    x_center, y_center, box_width, box_height = map(float, fields[1:])
    left = (x_center - box_width / 2) * width
    right = (x_center + box_width / 2) * width
    top = (y_center - box_height / 2) * height
    bottom = (y_center + box_height / 2) * height
    center_x, center_y = width / 2, height / 2
    radians = math.radians(angle_degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    corners = []
    for x, y in (
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ):
        dx, dy = x - center_x, y - center_y
        corners.append(
            (center_x + cosine * dx + sine * dy,
             center_y - sine * dx + cosine * dy)
        )
    clipped_left = max(0.0, min(point[0] for point in corners))
    clipped_right = min(float(width), max(point[0] for point in corners))
    clipped_top = max(0.0, min(point[1] for point in corners))
    clipped_bottom = min(float(height), max(point[1] for point in corners))
    if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
        return None
    return [
        class_id,
        format((clipped_left + clipped_right) / (2 * width), ".17g"),
        format((clipped_top + clipped_bottom) / (2 * height), ".17g"),
        format((clipped_right - clipped_left) / width, ".17g"),
        format((clipped_bottom - clipped_top) / height, ".17g"),
    ]


def image_fill(image: Image.Image) -> tuple[Image.Image, Any]:
    if image.mode == "RGB":
        return image, (114, 114, 114)
    if image.mode == "RGBA":
        return image, (114, 114, 114, 0)
    if image.mode == "L":
        return image, 114
    return image.convert("RGB"), (114, 114, 114)


def rotate_training_split() -> dict[str, float]:
    rng = random.Random(SEED)
    angles: dict[str, float] = {}
    for source_image in image_files(SOURCE_ROOT / "train" / "images"):
        angle = rng.uniform(-DEGREES, DEGREES)
        angles[source_image.name] = angle
        target_image = STAGING_ROOT / "train" / "images" / source_image.name
        with Image.open(source_image) as source:
            image, fill = image_fill(source)
            rotated = image.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=fill,
            )
            image_format = source.format or target_image.suffix.lstrip(".").upper()
            rotated.save(target_image, format=image_format)

        source_label = SOURCE_ROOT / "train" / "labels" / f"{source_image.stem}.txt"
        rows = [
            line.split()
            for line in source_label.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        with Image.open(source_image) as source:
            width, height = source.size
        rotated_rows = [
            row
            for fields in rows
            if (row := rotate_box(fields, width, height, angle)) is not None
        ]
        target_label = STAGING_ROOT / "train" / "labels" / f"{source_image.stem}.txt"
        target_label.write_text(
            "".join(" ".join(row) + "\n" for row in rotated_rows),
            encoding="utf-8",
            newline="\n",
        )
    return angles


def write_reports(report: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "M2 ROTATION DATASET AUDIT",
        f"Dataset: {report['dataset']}",
        f"Source dataset: {report['source_dataset']}",
        f"Rotation range: [-{DEGREES}, +{DEGREES}] degrees",
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
            *(f"  {key}: {len(items)}" for key, items in report["findings"].items()),
            f"TOTAL ERRORS: {report['error_count']}",
            f"PASSED: {str(report['passed']).lower()}",
        ]
    )
    if not report["passed"]:
        lines.extend(["", "FINDINGS:", json.dumps(report["findings"], indent=2)])
    AUDIT_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_dataset() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite M2 dataset: {OUTPUT_ROOT}")
    if STAGING_ROOT.exists():
        raise FileExistsError(f"Staging directory already exists: {STAGING_ROOT}")
    if not (SOURCE_ROOT / "data.yaml").is_file():
        raise FileNotFoundError(f"Source dataset YAML not found: {SOURCE_ROOT / 'data.yaml'}")

    source_config = yaml.safe_load((SOURCE_ROOT / "data.yaml").read_text(encoding="utf-8"))
    names = source_config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names != CLASS_NAMES:
        raise ValueError(f"Unexpected source class names: {names!r}")
    counts = source_finding_report()
    if counts:
        raise ValueError(f"Source split counts are invalid; stopping: {counts}")

    source_audit = audit_tree(SOURCE_ROOT, verify_val_test_source=False)
    write_reports(source_audit)
    if not source_audit["passed"]:
        raise RuntimeError(
            f"Source dataset audit failed; no M2 dataset was created. See {AUDIT_JSON}"
        )

    STAGING_ROOT.mkdir(parents=True)
    try:
        for split in EXPECTED_COUNTS:
            (STAGING_ROOT / split / "images").mkdir(parents=True)
            (STAGING_ROOT / split / "labels").mkdir(parents=True)
        angles = rotate_training_split()
        for split in ("val", "test"):
            for source_image in image_files(SOURCE_ROOT / split / "images"):
                shutil.copy2(
                    source_image,
                    STAGING_ROOT / split / "images" / source_image.name,
                )
                source_label = SOURCE_ROOT / split / "labels" / f"{source_image.stem}.txt"
                shutil.copy2(
                    source_label,
                    STAGING_ROOT / split / "labels" / source_label.name,
                )

        data_config = {
            "path": str(OUTPUT_ROOT.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 2,
            "names": CLASS_NAMES,
        }
        (STAGING_ROOT / "data.yaml").write_text(
            yaml.safe_dump(data_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        (STAGING_ROOT / "dataset_info.json").write_text(
            json.dumps(
                {
                    "experiment": "M2_rotation",
                    "source_dataset": str(SOURCE_ROOT.resolve()),
                    "seed": SEED,
                    "rotation": {
                        "degrees": DEGREES,
                        "sampled_range": [-DEGREES, DEGREES],
                        "one_rotated_image_per_source_train_image": True,
                        "angles_degrees": angles,
                    },
                    "training_originals_included": False,
                    "horizontal_flip": False,
                    "validation_test_augmentation": False,
                    "bbox_transform": "rotate all four box corners about image center, form the axis-aligned envelope, and clip to image bounds",
                    "epsilon_normalized": EPSILON,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        STAGING_ROOT.rename(OUTPUT_ROOT)
    except Exception:
        if STAGING_ROOT.exists():
            shutil.rmtree(STAGING_ROOT)
        raise

    report = audit_tree(OUTPUT_ROOT, verify_val_test_source=True)
    write_reports(report)
    print(f"M2_AUDIT_JSON: {AUDIT_JSON.resolve()}")
    print(f"M2_AUDIT_TEXT: {AUDIT_TEXT.resolve()}")
    print(f"M2_DATASET: {OUTPUT_ROOT.resolve()}")
    print(f"TRAIN_IMAGES: {report['split_counts']['train']['images']}")
    print(f"VAL_IMAGES: {report['split_counts']['val']['images']}")
    print(f"TEST_IMAGES: {report['split_counts']['test']['images']}")
    print(f"AUDIT_PASSED: {str(report['passed']).lower()}")
    if not report["passed"]:
        raise RuntimeError(f"M2 dataset audit failed; training must not start: {AUDIT_JSON}")


if __name__ == "__main__":
    create_dataset()
