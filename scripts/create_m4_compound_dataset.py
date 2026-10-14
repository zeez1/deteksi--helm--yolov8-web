#!/usr/bin/env python3
"""Create the M4 compound-augmentation dataset from the source splits."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "dataset" / "helmet_research"
OUTPUT_ROOT = ROOT / "dataset" / "experiments" / "M4_compound"
STAGING_ROOT = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.staging")
SEED = 42
ROTATION_DEGREES = 15.0
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


def validate_bbox(
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    location: str,
) -> None:
    if class_id not in CLASS_NAMES:
        raise ValueError(f"Invalid class id at {location}: {class_id}")
    values = (x_center, y_center, width, height)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite bounding box at {location}: {values}")
    if not (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    ):
        raise ValueError(f"Out-of-range bounding box at {location}: {values}")

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
        raise ValueError(
            f"Bounding box extends beyond image at {location}: "
            f"{(left, top, right, bottom)}"
        )


def read_label_rows(label_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"Expected five YOLO fields at {label_path}:{line_number}"
            )
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}") from exc
        validate_bbox(
            class_id,
            x_center,
            y_center,
            width,
            height,
            f"{label_path}:{line_number}",
        )
        rows.append(fields)
    if not rows:
        raise ValueError(f"Empty YOLO label file: {label_path}")
    return rows


def transform_box(
    fields: list[str],
    image_width: int,
    image_height: int,
    angle_degrees: float,
    location: str,
) -> list[str] | None:
    """Apply horizontal flip, then M2's clipped rotated-box transform."""
    class_id = int(fields[0])
    x_center, y_center, box_width, box_height = map(float, fields[1:])
    validate_bbox(class_id, x_center, y_center, box_width, box_height, location)

    flipped_x = 1.0 - x_center
    left = (flipped_x - box_width / 2) * image_width
    right = (flipped_x + box_width / 2) * image_width
    top = (y_center - box_height / 2) * image_height
    bottom = (y_center + box_height / 2) * image_height
    center_x, center_y = image_width / 2, image_height / 2
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
            (
                center_x + cosine * dx + sine * dy,
                center_y - sine * dx + cosine * dy,
            )
        )

    clipped_left = max(0.0, min(point[0] for point in corners))
    clipped_right = min(float(image_width), max(point[0] for point in corners))
    clipped_top = max(0.0, min(point[1] for point in corners))
    clipped_bottom = min(float(image_height), max(point[1] for point in corners))
    if clipped_right <= clipped_left or clipped_bottom <= clipped_top:
        return None

    transformed = [
        str(class_id),
        format((clipped_left + clipped_right) / (2 * image_width), ".17g"),
        format((clipped_top + clipped_bottom) / (2 * image_height), ".17g"),
        format((clipped_right - clipped_left) / image_width, ".17g"),
        format((clipped_bottom - clipped_top) / image_height, ".17g"),
    ]
    validate_bbox(
        class_id,
        *map(float, transformed[1:]),
        location,
    )
    return transformed


def image_for_rotation(image: Image.Image) -> tuple[Image.Image, Any]:
    if image.mode == "RGB":
        return image, (114, 114, 114)
    if image.mode == "RGBA":
        return image, (114, 114, 114, 0)
    if image.mode == "L":
        return image, 114
    return image.convert("RGB"), (114, 114, 114)


def validate_source() -> dict[str, Any]:
    config_path = SOURCE_ROOT / "data.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Source dataset YAML not found: {config_path}")
    source_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(source_config, dict):
        raise ValueError(f"Source dataset YAML must be a mapping: {config_path}")
    names = source_config.get("names")
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names != CLASS_NAMES or source_config.get("nc") != len(CLASS_NAMES):
        raise ValueError(f"Unexpected source class configuration: {names!r}")

    splits: dict[str, list[Path]] = {}
    for split, expected in EXPECTED_COUNTS.items():
        images = image_files(SOURCE_ROOT / split / "images")
        labels_dir = SOURCE_ROOT / split / "labels"
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Label directory not found: {labels_dir}")
        labels = sorted(labels_dir.glob("*.txt"), key=lambda path: path.name.lower())
        image_stems = [path.stem for path in images]
        label_stems = [path.stem for path in labels]
        if (
            len(images) != expected
            or len(labels) != expected
            or len(set(image_stems)) != len(image_stems)
            or set(image_stems) != set(label_stems)
        ):
            raise ValueError(
                f"Unexpected {split} image/label structure: "
                f"{len(images)} images, {len(labels)} labels"
            )
        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                raise ValueError(f"Corrupt source image {image_path}: {exc}") from exc
            read_label_rows(labels_dir / f"{image_path.stem}.txt")
        splits[split] = images
    return {"config": source_config, "splits": splits}


def write_transformed_train(
    source_images: list[Path],
    destination: Path,
) -> tuple[dict[str, float], int]:
    image_dir = destination / "train" / "images"
    label_dir = destination / "train" / "labels"
    rng = random.Random(SEED)
    angles: dict[str, float] = {}
    dropped_objects = 0

    for source_image in source_images:
        angle = rng.uniform(-ROTATION_DEGREES, ROTATION_DEGREES)
        angles[source_image.name] = angle
        source_label = SOURCE_ROOT / "train" / "labels" / f"{source_image.stem}.txt"
        source_rows = read_label_rows(source_label)
        target_image = image_dir / source_image.name
        with Image.open(source_image) as original:
            width, height = original.size
            mirrored = ImageOps.mirror(original)
            rotatable, fill = image_for_rotation(mirrored)
            rotated = rotatable.rotate(
                angle,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=fill,
            )
            brightened = ImageEnhance.Brightness(rotated).enhance(
                BRIGHTNESS_FACTOR
            )
            adjusted = ImageEnhance.Contrast(brightened).enhance(CONTRAST_FACTOR)
            if target_image.suffix.lower() in {".jpg", ".jpeg"} and adjusted.mode not in {
                "RGB",
                "L",
            }:
                adjusted = adjusted.convert("RGB")
            adjusted.save(
                target_image,
                format=original.format or target_image.suffix.lstrip(".").upper(),
            )

        transformed_rows = [
            row
            for fields in source_rows
            if (
                row := transform_box(
                    fields,
                    width,
                    height,
                    angle,
                    f"{source_label}:{source_image.name}",
                )
            )
            is not None
        ]
        if not transformed_rows:
            raise ValueError(
                f"All objects were cropped from transformed image {source_image.name}"
            )
        dropped_objects += len(source_rows) - len(transformed_rows)
        (label_dir / source_label.name).write_text(
            "".join(" ".join(row) + "\n" for row in transformed_rows),
            encoding="utf-8",
            newline="\n",
        )
    return angles, dropped_objects


def copy_unchanged_split(split: str, source_images: list[Path], destination: Path) -> None:
    for source_image in source_images:
        shutil.copy2(source_image, destination / split / "images" / source_image.name)
        source_label = SOURCE_ROOT / split / "labels" / f"{source_image.stem}.txt"
        shutil.copy2(source_label, destination / split / "labels" / source_label.name)


def split_metadata(
    root: Path, split: str
) -> tuple[dict[str, Any], Counter[int]]:
    images = image_files(root / split / "images")
    labels = sorted((root / split / "labels").glob("*.txt"))
    objects: Counter[int] = Counter()
    for label in labels:
        for row in read_label_rows(label):
            objects[int(row[0])] += 1
    return (
        {
            "images": len(images),
            "labels": len(labels),
            "objects": {
                "helmet": objects[0],
                "no-helmet": objects[1],
            },
        },
        objects,
    )


def create_dataset() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite M4 dataset: {OUTPUT_ROOT}")
    if STAGING_ROOT.exists():
        raise FileExistsError(f"Staging directory already exists: {STAGING_ROOT}")

    source = validate_source()
    source_splits: dict[str, list[Path]] = source["splits"]
    STAGING_ROOT.mkdir(parents=True)
    try:
        for split in EXPECTED_COUNTS:
            (STAGING_ROOT / split / "images").mkdir(parents=True)
            (STAGING_ROOT / split / "labels").mkdir(parents=True)

        angles, dropped_objects = write_transformed_train(
            source_splits["train"], STAGING_ROOT
        )
        for split in ("val", "test"):
            copy_unchanged_split(split, source_splits[split], STAGING_ROOT)

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

        counts: dict[str, Any] = {}
        for split in EXPECTED_COUNTS:
            counts[split], _ = split_metadata(STAGING_ROOT, split)
            if (
                counts[split]["images"] != EXPECTED_COUNTS[split]
                or counts[split]["labels"] != EXPECTED_COUNTS[split]
            ):
                raise ValueError(f"Unexpected generated {split} counts: {counts[split]}")

        for split in ("val", "test"):
            for source_image in source_splits[split]:
                copied_image = STAGING_ROOT / split / "images" / source_image.name
                source_label = SOURCE_ROOT / split / "labels" / f"{source_image.stem}.txt"
                copied_label = STAGING_ROOT / split / "labels" / source_label.name
                if (
                    sha256_file(source_image) != sha256_file(copied_image)
                    or sha256_file(source_label) != sha256_file(copied_label)
                ):
                    raise ValueError(
                        f"{split} files were not copied unchanged: {source_image.name}"
                    )

        class_object_counts = {
            "helmet": sum(counts[split]["objects"]["helmet"] for split in counts),
            "no-helmet": sum(
                counts[split]["objects"]["no-helmet"] for split in counts
            ),
        }
        metadata = {
            "experiment": "M4_compound",
            "source_dataset": str(SOURCE_ROOT.resolve()),
            "seed": SEED,
            "train_images": counts["train"]["images"],
            "train_labels": counts["train"]["labels"],
            "val_images": counts["val"]["images"],
            "val_labels": counts["val"]["labels"],
            "test_images": counts["test"]["images"],
            "test_labels": counts["test"]["labels"],
            "split_counts": counts,
            "class_names": CLASS_NAMES,
            "class_object_counts": class_object_counts,
            "augmentation": {
                "type": "compound_offline",
                "applied_to": "train only; one compound-transformed image per source image",
                "sequence": [
                    "horizontal_flip",
                    "rotation",
                    "brightness",
                    "contrast",
                ],
                "horizontal_flip": True,
                "rotation_degrees_range": [
                    -ROTATION_DEGREES,
                    ROTATION_DEGREES,
                ],
                "rotation_seed": SEED,
                "brightness_factor": BRIGHTNESS_FACTOR,
                "contrast_factor": CONTRAST_FACTOR,
                "other_augmentation": False,
                "bounding_box_transform": (
                    "horizontal flip; rotate four corners; axis-aligned envelope; "
                    "clip to image bounds"
                ),
                "dropped_cropped_objects": dropped_objects,
                "rotation_angle_by_image_degrees": angles,
            },
            "validation_test_copied_unchanged": True,
            "label_format": "YOLO normalized xywh",
        }
        (STAGING_ROOT / "dataset_info.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        STAGING_ROOT.rename(OUTPUT_ROOT)
    except Exception:
        shutil.rmtree(STAGING_ROOT)
        raise

    print(f"M4_DATASET: {OUTPUT_ROOT.resolve()}")
    for split, details in counts.items():
        print(
            f"{split.upper()}: images={details['images']} labels={details['labels']} "
            f"helmet={details['objects']['helmet']} "
            f"no-helmet={details['objects']['no-helmet']}"
        )
    print("VALIDATION_TEST_IDENTICAL: true")


if __name__ == "__main__":
    create_dataset()
