#!/usr/bin/env python3
"""Audit the source YOLO dataset and create reproducible, unaugmented splits."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import cv2
from PIL import Image, UnidentifiedImageError


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = {0: "helmet", 1: "no-helmet"}
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
SOURCE_ZIP = Path(
    "dataset/yolov8-helmet-final-use-no-augmentation.v2-no-augmentasi.yolov8.zip"
)
SOURCE_DIR = Path("dataset/source")
RESEARCH_DIR = Path("dataset/helmet_research")
REPORT_DIR = Path("reports")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_label_for(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def ensure_source_extracted(archive: Path, source: Path) -> None:
    if source.exists() and any(source.iterdir()):
        return
    if not archive.is_file():
        raise FileNotFoundError(f"Source ZIP not found: {archive}")
    source.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            member = PurePosixPath(entry.filename)
            windows_member = PureWindowsPath(entry.filename)
            if (
                member.is_absolute()
                or windows_member.is_absolute()
                or windows_member.drive
                or "\\" in entry.filename
                or ".." in member.parts
            ):
                raise ValueError(f"Unsafe ZIP member path: {entry.filename}")
            target = source.joinpath(*member.parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(entry) as input_stream, target.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)


def image_label_maps(
    source: Path,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]], dict[str, dict[str, int]]]:
    images_by_split: dict[str, list[Path]] = defaultdict(list)
    labels_by_split: dict[str, list[Path]] = defaultdict(list)
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        split = relative.parts[0] if len(relative.parts) > 1 else "root"
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images_by_split[split].append(path)
        elif path.suffix.lower() == ".txt" and path.parent.name.lower() == "labels":
            labels_by_split[split].append(path)

    image_lookup: dict[str, dict[str, int]] = defaultdict(dict)
    label_lookup: dict[str, dict[str, int]] = defaultdict(dict)
    for split, paths in images_by_split.items():
        for path in paths:
            image_lookup[split][path.stem] = image_lookup[split].get(path.stem, 0) + 1
    for split, paths in labels_by_split.items():
        for path in paths:
            label_lookup[split][path.stem] = label_lookup[split].get(path.stem, 0) + 1
    return images_by_split, labels_by_split, {
        "image_duplicates_by_stem": image_lookup,
        "label_duplicates_by_stem": label_lookup,
    }


def audit_dataset(source: Path) -> dict[str, Any]:
    images_by_split, labels_by_split, stem_maps = image_label_maps(source)
    images = sorted(
        (path for paths in images_by_split.values() for path in paths),
        key=lambda path: path.as_posix().lower(),
    )
    labels = sorted(
        (path for paths in labels_by_split.values() for path in paths),
        key=lambda path: path.as_posix().lower(),
    )
    issues: dict[str, list[dict[str, str]]] = {
        key: []
        for key in (
            "corrupt_images",
            "unreadable_by_opencv",
            "unreadable_by_pil",
            "label_parse_errors",
            "invalid_class_ids",
            "x_center_out_of_range",
            "y_center_out_of_range",
            "width_out_of_range",
            "height_out_of_range",
            "nonpositive_width",
            "nonpositive_height",
            "images_without_labels",
            "labels_without_images",
            "duplicate_image_hashes",
            "duplicate_image_stems",
            "duplicate_label_stems",
            "empty_dataset",
        )
    }
    for map_name, issue_name in (
        ("image_duplicates_by_stem", "duplicate_image_stems"),
        ("label_duplicates_by_stem", "duplicate_label_stems"),
    ):
        for split, counts_by_stem in stem_maps[map_name].items():
            for stem, count in counts_by_stem.items():
                if count > 1:
                    issues.setdefault(issue_name, []).append(
                        {"file": f"{split}/{stem}", "count": str(count)}
                    )
    if not images:
        issues["empty_dataset"].append({"directory": str(source)})
    class_counts: Counter[int] = Counter()
    empty_labels: list[str] = []
    image_formats: Counter[str] = Counter()
    resolutions: list[dict[str, Any]] = []
    hashes: dict[str, list[Path]] = defaultdict(list)
    readable_images: dict[str, dict[str, Any]] = {}
    fatal_issue_keys = (
        "corrupt_images",
        "unreadable_by_opencv",
        "unreadable_by_pil",
        "label_parse_errors",
        "invalid_class_ids",
        "x_center_out_of_range",
        "y_center_out_of_range",
        "width_out_of_range",
        "height_out_of_range",
        "nonpositive_width",
        "nonpositive_height",
        "images_without_labels",
        "labels_without_images",
        "duplicate_image_stems",
        "duplicate_label_stems",
        "empty_dataset",
    )

    for image_path in images:
        relative = image_path.relative_to(source).as_posix()
        try:
            with Image.open(image_path) as image:
                image_format = image.format or "UNKNOWN"
                width, height = image.size
                image.verify()
            with Image.open(image_path) as image:
                image.load()
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            issues["corrupt_images"].append(
                {"file": relative, "error": f"{type(exc).__name__}: {exc}"}
            )
            issues["unreadable_by_pil"].append({"file": relative})
            continue

        cv_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if cv_image is None:
            issues["unreadable_by_opencv"].append({"file": relative})
        image_formats[image_format] += 1
        resolutions.append({"file": relative, "width": width, "height": height})
        hashes[sha256_file(image_path)].append(image_path)
        readable_images[relative] = {
            "split": image_path.relative_to(source).parts[0],
            "stem": image_path.stem,
        }

    for digest, paths in hashes.items():
        if len(paths) > 1:
            issues["duplicate_image_hashes"].append(
                {
                    "sha256": digest,
                    "files": [path.relative_to(source).as_posix() for path in paths],
                }
            )

    image_split_stems: dict[str, set[str]] = defaultdict(set)
    label_split_stems: dict[str, set[str]] = defaultdict(set)
    for path in images:
        relative = path.relative_to(source)
        split = relative.parts[0] if len(relative.parts) > 1 else "root"
        image_split_stems[split].add(path.stem)
    for path in labels:
        relative = path.relative_to(source)
        split = relative.parts[0] if len(relative.parts) > 1 else "root"
        label_split_stems[split].add(path.stem)
        if path.stat().st_size == 0 or not path.read_text(encoding="utf-8-sig").strip():
            empty_labels.append(relative.as_posix())
            continue

        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            location = {"file": relative.as_posix(), "line": str(line_number)}
            if len(fields) != 5:
                issues["label_parse_errors"].append(
                    {**location, "error": f"Expected 5 fields, found {len(fields)}"}
                )
                continue
            try:
                class_value = float(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError:
                issues["label_parse_errors"].append(
                    {**location, "error": "Class ID or coordinate is not numeric"}
                )
                continue
            if not math.isfinite(class_value) or not class_value.is_integer():
                issues["invalid_class_ids"].append(
                    {**location, "class_id": fields[0], "error": "Class ID must be an integer"}
                )
            else:
                class_id = int(class_value)
                if class_id not in CLASS_NAMES:
                    issues["invalid_class_ids"].append(
                        {**location, "class_id": fields[0], "error": "Only class IDs 0 and 1 are valid"}
                    )
                else:
                    class_counts[class_id] += 1

            if not all(math.isfinite(value) for value in values):
                issues["label_parse_errors"].append(
                    {**location, "error": "Coordinates must be finite numbers"}
                )
                continue
            x_center, y_center, width, height = values
            for name, value, key in (
                ("x_center", x_center, "x_center_out_of_range"),
                ("y_center", y_center, "y_center_out_of_range"),
                ("width", width, "width_out_of_range"),
                ("height", height, "height_out_of_range"),
            ):
                if value < 0 or value > 1:
                    issues[key].append(
                        {**location, "value": str(value), "field": name}
                    )
            if width <= 0:
                issues["nonpositive_width"].append({**location, "value": str(width)})
            if height <= 0:
                issues["nonpositive_height"].append({**location, "value": str(height)})

    for split in sorted(set(image_split_stems) | set(label_split_stems)):
        for stem in sorted(image_split_stems[split] - label_split_stems[split]):
            matches = [path for path in images_by_split[split] if path.stem == stem]
            issues["images_without_labels"].extend(
                {"file": path.relative_to(source).as_posix()} for path in matches
            )
        for stem in sorted(label_split_stems[split] - image_split_stems[split]):
            matches = [path for path in labels_by_split[split] if path.stem == stem]
            issues["labels_without_images"].extend(
                {"file": path.relative_to(source).as_posix()} for path in matches
            )

    split_counts = {
        split: {
            "images": len(paths),
            "labels": len(labels_by_split.get(split, [])),
        }
        for split, paths in sorted(images_by_split.items())
    }
    minimum = min(resolutions, key=lambda item: item["width"] * item["height"]) if resolutions else None
    maximum = max(resolutions, key=lambda item: item["width"] * item["height"]) if resolutions else None
    duplicate_count = sum(
        len(item["files"]) - 1 for item in issues["duplicate_image_hashes"]
    )
    return {
        "source_directory": str(source.resolve()),
        "total_images": len(images),
        "total_labels": len(labels),
        "source_splits": split_counts,
        "class_distribution": [
            {"class_id": class_id, "class_name": class_name,
             "object_count": class_counts[class_id]}
            for class_id, class_name in CLASS_NAMES.items()
        ],
        "empty_label_count": len(empty_labels),
        "empty_labels": empty_labels,
        "image_formats": dict(sorted(image_formats.items())),
        "resolution_minimum": minimum,
        "resolution_maximum": maximum,
        "duplicate_image_hash_groups": issues["duplicate_image_hashes"],
        "duplicate_image_count_beyond_first": duplicate_count,
        "issues": issues,
        "fatal_issue_count": sum(len(issues[key]) for key in fatal_issue_keys),
        "_readable_images": readable_images,
    }


def audit_text(report: dict[str, Any]) -> str:
    lines = [
        "DATASET SOURCE AUDIT",
        f"Source: {report['source_directory']}",
        f"Total images: {report['total_images']}",
        f"Total labels: {report['total_labels']}",
        "Source splits:",
    ]
    for split, counts in report["source_splits"].items():
        lines.append(f"  {split}: {counts['images']} images, {counts['labels']} labels")
    lines.extend(["Class distribution:"])
    for item in report["class_distribution"]:
        lines.append(
            f"  {item['class_id']} ({item['class_name']}): {item['object_count']} objects"
        )
    lines.extend(
        [
            f"Empty labels (informational): {report['empty_label_count']}",
            f"Image formats: {json.dumps(report['image_formats'], ensure_ascii=False)}",
            f"Minimum resolution: {report['resolution_minimum']}",
            f"Maximum resolution: {report['resolution_maximum']}",
            f"Duplicate image groups (SHA-256): {len(report['duplicate_image_hash_groups'])}",
            f"Duplicate image count beyond first: {report['duplicate_image_count_beyond_first']}",
            f"Fatal issue count: {report['fatal_issue_count']}",
            "",
            "Findings:",
        ]
    )
    for issue_name, entries in report["issues"].items():
        lines.append(f"  {issue_name}: {len(entries)}")
        for entry in entries:
            lines.append(f"    {json.dumps(entry, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def write_class_csv(path: Path, distribution: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["class_id,class_name,object_count"]
    lines.extend(
        f"{item['class_id']},{item['class_name']},{item['object_count']}"
        for item in distribution
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_splits(report: dict[str, Any], source: Path, target: Path) -> dict[str, Any]:
    image_records = report["_readable_images"]
    split_groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(
        (source / relative for relative in image_records),
        key=lambda item: item.as_posix().lower(),
    ):
        split_groups[sha256_file(path)].append(path)

    groups = list(split_groups.values())
    random.Random(SEED).shuffle(groups)
    total = report["total_images"]
    targets = {"train": int(total * 0.70), "val": int(total * 0.20)}
    targets["test"] = total - targets["train"] - targets["val"]
    assigned = {"train": [], "val": [], "test": []}
    counts = {key: 0 for key in assigned}
    for group in groups:
        # Keep byte-identical images together and fill the largest remaining split first.
        fitting_splits = [
            name for name in assigned if counts[name] + len(group) <= targets[name]
        ]
        candidates = fitting_splits or list(assigned)
        if fitting_splits:
            split = max(candidates, key=lambda name: targets[name] - counts[name])
        else:
            split = min(
                candidates,
                key=lambda name: abs((counts[name] + len(group)) - targets[name]),
            )
        assigned[split].extend(group)
        counts[split] += len(group)

    if target.exists():
        raise FileExistsError(
            f"Refusing to write dataset: destination already exists: {target}"
        )
    for split, split_images in assigned.items():
        (target / split / "images").mkdir(parents=True, exist_ok=True)
        (target / split / "labels").mkdir(parents=True, exist_ok=True)
        for image in split_images:
            shutil.copy2(image, target / split / "images" / image.name)
            source_label = source_label_for(image)
            shutil.copy2(source_label, target / split / "labels" / source_label.name)

    data_yaml = (
        f"path: {target.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 2\n\n"
        "names:\n"
        "  0: helmet\n"
        "  1: no-helmet\n"
    )
    (target / "data.yaml").write_text(data_yaml, encoding="utf-8")

    info = {
        "generator": "scripts/prepare_dataset.py",
        "seed": SEED,
        "source_archive_sha256": report["archive_sha256"],
        "source_directory": str(source.resolve()),
        "research_directory": str(target.resolve()),
        "source_image_count": total,
        "split_targets": targets,
        "split_counts": counts,
        "split_percentages": {
            split: round(count / total * 100, 4) if total else 0
            for split, count in counts.items()
        },
        "class_names": CLASS_NAMES,
        "augmentation": "none",
        "image_sha256_by_split": {
            split: sorted(sha256_file(path) for path in split_images)
            for split, split_images in assigned.items()
        },
    }
    write_json(target / "dataset_info.json", info)
    return info


def validate_splits(target: Path) -> dict[str, Any]:
    counts: dict[str, dict[str, Any]] = {}
    all_hashes: dict[str, set[str]] = {}
    mismatches: list[dict[str, str]] = []
    invalid_classes: list[dict[str, str]] = []
    class_counts: Counter[int] = Counter()
    for split in ("train", "val", "test"):
        image_dir = target / split / "images"
        label_dir = target / split / "labels"
        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        labels = sorted(path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        for stem in sorted(image_stems - label_stems):
            mismatches.append({"split": split, "image_without_label": stem})
        for stem in sorted(label_stems - image_stems):
            mismatches.append({"split": split, "label_without_image": stem})
        current_hashes: set[str] = set()
        for image in images:
            digest = sha256_file(image)
            current_hashes.add(digest)
            all_hashes.setdefault(digest, set()).add(split)
        for label in labels:
            for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), start=1):
                if not line.strip():
                    continue
                fields = line.split()
                try:
                    class_value = float(fields[0])
                    class_id = int(class_value)
                except (ValueError, IndexError, OverflowError):
                    invalid_classes.append({"split": split, "file": label.name, "line": str(line_number)})
                    continue
                if not class_value.is_integer() or class_id not in CLASS_NAMES:
                    invalid_classes.append(
                        {"split": split, "file": label.name, "line": str(line_number), "class_id": fields[0]}
                    )
                    continue
                class_counts[class_id] += 1
        total_images = sum(
            1 for current_split in ("train", "val", "test")
            for path in (target / current_split / "images").iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[split] = {
            "images": len(images),
            "labels": len(labels),
            "percentage": round(len(images) / total_images * 100, 4) if total_images else 0,
        }

    duplicate_groups = {
        digest: sorted(splits) for digest, splits in all_hashes.items() if len(splits) > 1
    }
    return {
        "split_counts": counts,
        "class_distribution": [
            {"class_id": class_id, "class_name": name, "object_count": class_counts[class_id]}
            for class_id, name in CLASS_NAMES.items()
        ],
        "image_label_mismatches": mismatches,
        "duplicate_groups_across_splits": duplicate_groups,
        "invalid_class_ids": invalid_classes,
        "validation_passed": not mismatches and not duplicate_groups and not invalid_classes,
    }


def split_text(result: dict[str, Any]) -> str:
    lines = ["FINAL RESEARCH DATASET SPLIT REPORT", f"Validation passed: {result['validation_passed']}", "Splits:"]
    for split, counts in result["split_counts"].items():
        lines.append(
            f"  {split}: {counts['images']} images, {counts['labels']} labels "
            f"({counts['percentage']}%)"
        )
    lines.append("Class distribution:")
    for item in result["class_distribution"]:
        lines.append(f"  {item['class_id']} ({item['class_name']}): {item['object_count']} objects")
    lines.extend(
        [
            f"Image-label mismatches: {len(result['image_label_mismatches'])}",
            f"Duplicate groups across splits: {len(result['duplicate_groups_across_splits'])}",
            f"Invalid class IDs: {len(result['invalid_class_ids'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    archive = PROJECT_ROOT / SOURCE_ZIP
    source = PROJECT_ROOT / SOURCE_DIR
    target = PROJECT_ROOT / RESEARCH_DIR
    reports = PROJECT_ROOT / REPORT_DIR

    if target.exists():
        print(
            f"Warning: destination already exists; no dataset files were changed: {target}",
            file=sys.stderr,
        )
        return 4
    ensure_source_extracted(archive, source)
    report = audit_dataset(source)
    report["archive"] = str(archive.resolve())
    report["archive_sha256"] = sha256_file(archive)
    write_class_csv(reports / "class_distribution.csv", report["class_distribution"])
    write_json_path = reports / "dataset_audit.json"
    json_report = {key: value for key, value in report.items() if not key.startswith("_")}
    write_json(write_json_path, json_report)
    (reports / "dataset_audit.txt").write_text(audit_text(json_report), encoding="utf-8")

    if report["fatal_issue_count"]:
        print(
            f"Audit found {report['fatal_issue_count']} integrity issue(s); "
            "reports saved. Research split was not created.",
            file=sys.stderr,
        )
        return 2

    info = create_splits(report, source, target)
    validation = validate_splits(target)
    split_report = {
        **validation,
        "seed": SEED,
        "augmentation": "none",
        "split_targets": info["split_targets"],
        "research_directory": str(target.resolve()),
        "data_yaml": str((target / "data.yaml").resolve()),
        "dataset_info": str((target / "dataset_info.json").resolve()),
    }
    write_json(reports / "final_split_report.json", split_report)
    (reports / "final_split_report.txt").write_text(split_text(split_report), encoding="utf-8")
    print(f"Audit: {reports / 'dataset_audit.txt'}")
    print(f"Split report: {reports / 'final_split_report.txt'}")
    print(f"Split validation passed: {validation['validation_passed']}")
    return 0 if validation["validation_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
