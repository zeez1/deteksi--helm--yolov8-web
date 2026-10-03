#!/usr/bin/env python3
"""Create the M1 dataset with every training image horizontally flipped."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent.parent
SOURCE_ROOT = ROOT / "dataset" / "helmet_research"
OUTPUT_ROOT = ROOT / "dataset" / "experiments" / "M1_flip"
STAGING_ROOT = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.staging")
SEED = 42
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
        raise FileNotFoundError(f"Dataset split directory not found: {directory}")
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def validate_source_split(split: str) -> list[Path]:
    images = image_files(SOURCE_ROOT / split / "images")
    labels_dir = SOURCE_ROOT / split / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Dataset labels directory not found: {labels_dir}")
    labels = sorted(path for path in labels_dir.glob("*.txt") if path.is_file())
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    if len(images) != EXPECTED_COUNTS[split]:
        raise ValueError(
            f"Expected {EXPECTED_COUNTS[split]} {split} images, found {len(images)}"
        )
    if image_stems != label_stems:
        raise ValueError(
            f"{split} image/label mismatch: "
            f"missing labels={sorted(image_stems - label_stems)[:5]}, "
            f"labels without images={sorted(label_stems - image_stems)[:5]}"
        )
    for label_path in labels:
        validate_label(label_path)
    return images


def validate_bbox(
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    location: str,
) -> None:
    if class_id not in CLASS_NAMES:
        raise ValueError(f"Invalid class id at {location}")
    if not (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    ):
        raise ValueError(f"Out-of-range box at {location}")

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
        raise ValueError(f"Box extends beyond image at {location}")


def flip_label_row(fields: list[str], location: str) -> list[str]:
    class_id = int(fields[0])
    x_center, y_center, width, height = map(float, fields[1:])
    validate_bbox(class_id, x_center, y_center, width, height, location)

    flipped_x_center = 1.0 - x_center
    flipped = [
        fields[0],
        format(flipped_x_center, ".17g"),
        fields[2],
        fields[3],
        fields[4],
    ]
    flipped_x_center, y_center, width, height = map(float, flipped[1:])
    validate_bbox(class_id, flipped_x_center, y_center, width, height, location)
    return flipped


def validate_label(label_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected 5 fields at {label_path}:{line_number}")
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid label at {label_path}:{line_number}") from exc
        validate_bbox(
            class_id,
            x_center,
            y_center,
            width,
            height,
            f"{label_path}:{line_number}",
        )
        rows.append(fields)
    return rows


def write_flipped_label(source: Path, target: Path) -> int:
    rows = validate_label(source)
    flipped_rows = [
        flip_label_row(fields, f"{source}:{line_number}")
        for line_number, fields in enumerate(rows, start=1)
    ]
    target.write_text(
        "".join(" ".join(fields) + "\n" for fields in flipped_rows),
        encoding="utf-8",
        newline="\n",
    )
    written = validate_label(target)
    if len(written) != len(rows):
        raise ValueError(f"Flipped label count changed: {source}")
    for original, flipped in zip(rows, written):
        if (
            original[0] != flipped[0]
            or abs(float(flipped[1]) - (1.0 - float(original[1]))) > 1e-7
            or original[2:] != flipped[2:]
        ):
            raise ValueError(f"Bounding-box transformation mismatch: {source}")
    return len(rows)


def copy_split(split: str, images: list[Path], destination: Path) -> None:
    target_images = destination / split / "images"
    target_labels = destination / split / "labels"
    target_images.mkdir(parents=True)
    target_labels.mkdir(parents=True)
    for image in images:
        shutil.copy2(image, target_images / image.name)
        shutil.copy2(
            SOURCE_ROOT / split / "labels" / f"{image.stem}.txt",
            target_labels / f"{image.stem}.txt",
        )


def create_dataset() -> None:
    random.seed(SEED)
    if not (SOURCE_ROOT / "data.yaml").is_file():
        raise FileNotFoundError(f"Source dataset YAML not found: {SOURCE_ROOT / 'data.yaml'}")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite M1 dataset: {OUTPUT_ROOT}")
    if STAGING_ROOT.exists():
        raise FileExistsError(f"Staging directory already exists: {STAGING_ROOT}")

    source_splits = {
        split: validate_source_split(split) for split in EXPECTED_COUNTS
    }
    source_config: dict[str, Any] = yaml.safe_load(
        (SOURCE_ROOT / "data.yaml").read_text(encoding="utf-8")
    )
    names = source_config.get("names")
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    if names != CLASS_NAMES:
        raise ValueError(f"Unexpected source class names: {names!r}")

    STAGING_ROOT.parent.mkdir(parents=True, exist_ok=True)
    STAGING_ROOT.mkdir()
    try:
        train_images_dir = STAGING_ROOT / "train" / "images"
        train_labels_dir = STAGING_ROOT / "train" / "labels"
        train_images_dir.mkdir(parents=True)
        train_labels_dir.mkdir(parents=True)
        box_count = 0
        class_counts: Counter[int] = Counter()
        for source_image in source_splits["train"]:
            target_image = train_images_dir / source_image.name
            with Image.open(source_image) as image:
                ImageOps.mirror(image).save(target_image, format=image.format)
            source_label = SOURCE_ROOT / "train" / "labels" / f"{source_image.stem}.txt"
            box_count += write_flipped_label(
                source_label,
                train_labels_dir / f"{source_image.stem}.txt",
            )
            class_counts.update(int(row[0]) for row in validate_label(source_label))

        for split in ("val", "test"):
            copy_split(split, source_splits[split], STAGING_ROOT)
            for image in source_splits[split]:
                label_path = SOURCE_ROOT / split / "labels" / f"{image.stem}.txt"
                class_counts.update(int(row[0]) for row in validate_label(label_path))

        data_config = {
            "path": str(OUTPUT_ROOT.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(CLASS_NAMES),
            "names": CLASS_NAMES,
        }
        (STAGING_ROOT / "data.yaml").write_text(
            yaml.safe_dump(data_config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        metadata = {
            "experiment": "M1_flip",
            "seed": SEED,
            "source_dataset": str(SOURCE_ROOT.resolve()),
            "horizontal_flip_probability": 1.0,
            "train_images": len(source_splits["train"]),
            "train_labels": len(source_splits["train"]),
            "train_boxes": box_count,
            "val_images": len(source_splits["val"]),
            "val_labels": len(source_splits["val"]),
            "test_images": len(source_splits["test"]),
            "test_labels": len(source_splits["test"]),
            "class_names": CLASS_NAMES,
            "class_object_counts": {
                "helmet": class_counts[0],
                "no-helmet": class_counts[1],
            },
            "augmentation": {
                "type": "horizontal_flip",
                "probability": 1.0,
                "applied_to": "train only; each source image replaced by its flip",
                "parameters": {
                    "horizontal_flip": True,
                    "vertical_flip": False,
                    "rotation": False,
                    "brightness_contrast": False,
                    "scale": False,
                    "mosaic": False,
                    "mixup": False,
                    "hsv": False,
                    "shear": False,
                    "perspective": False,
                    "translation": False,
                },
            },
            "validation_and_test_copied_unchanged": True,
            "label_format": "YOLO normalized xywh; x_center transformed to 1 - x_center",
        }
        (STAGING_ROOT / "dataset_info.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for split in ("val", "test"):
            for source_file in sorted(
                (SOURCE_ROOT / split).rglob("*"),
                key=lambda path: path.as_posix().lower(),
            ):
                if source_file.is_file():
                    copied_file = STAGING_ROOT / source_file.relative_to(SOURCE_ROOT)
                    if not copied_file.is_file() or sha256_file(source_file) != sha256_file(
                        copied_file
                    ):
                        raise ValueError(f"{split} file was not copied identically: {source_file}")

        STAGING_ROOT.rename(OUTPUT_ROOT)
    except Exception:
        shutil.rmtree(STAGING_ROOT)
        raise

    print(f"M1_DATASET: {OUTPUT_ROOT.resolve()}")
    print(f"TRAIN_IMAGES: {len(source_splits['train'])}")
    print(f"TRAIN_BOXES: {box_count}")
    print(f"VAL_IMAGES: {len(source_splits['val'])}")
    print(f"TEST_IMAGES: {len(source_splits['test'])}")
    print("VAL_TEST_IDENTICAL: true")


if __name__ == "__main__":
    create_dataset()
