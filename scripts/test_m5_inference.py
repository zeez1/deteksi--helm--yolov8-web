#!/usr/bin/env python3
"""Run M5 object detection on one image and save an annotated copy."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "experiments" / "M5_controlled_augmentation" / "weights" / "best.pt"
OUTPUT_ROOT = ROOT / "inference" / "M5"
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to one image to run inference on",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detection",
    )
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error("--conf must be greater than 0 and less than or equal to 1")
    return args


def class_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        result = {int(class_id): str(name) for class_id, name in names.items()}
    elif isinstance(names, (list, tuple)):
        result = {class_id: str(name) for class_id, name in enumerate(names)}
    else:
        raise TypeError(f"Unsupported model.names value: {type(names).__name__}")
    if not result:
        raise ValueError("The loaded M5 model has no class names")
    return result


def unique_run_name() -> str:
    return f"run_{uuid.uuid4().hex}"


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Source must be an image file: {source}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"M5 model weights not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    names = class_names(model.names)
    names_by_normalized = {
        name.strip().casefold(): (class_id, name)
        for class_id, name in names.items()
    }
    required_names = ("helmet", "no-helmet")
    missing_names = [
        name for name in required_names if name not in names_by_normalized
    ]
    if missing_names:
        raise ValueError(
            f"M5 model.names is missing expected classes {missing_names}; "
            f"found {names}"
        )

    device: int | str = 0 if torch.cuda.is_available() else "cpu"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_name = unique_run_name()
    while (OUTPUT_ROOT / run_name).exists():
        run_name = unique_run_name()

    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=640,
        device=device,
        save=True,
        project=str(OUTPUT_ROOT),
        name=run_name,
        exist_ok=False,
        verbose=False,
    )
    if not results:
        raise RuntimeError(f"No inference result returned for image: {source}")

    result = results[0]
    counts = {class_id: 0 for class_id in names}
    detections: list[tuple[str, float, list[float]]] = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id not in names:
                raise ValueError(
                    f"Prediction contains unknown class ID {class_id}; "
                    f"model.names={names}"
                )
            counts[class_id] += 1
            detections.append(
                (
                    names[class_id],
                    float(box.conf.item()),
                    [float(value) for value in box.xyxy[0].tolist()],
                )
            )

    output_dir = Path(result.save_dir).resolve()
    output_path = output_dir / source.name
    if not output_path.is_file():
        raise FileNotFoundError(
            f"Annotated inference image was not saved where expected: {output_path}"
        )

    print(f"Model: {MODEL_PATH.resolve()}")
    print(f"Class names (model.names): {names}")
    print(f"Device: {device}")
    print(f"Source: {source}")
    print(f"Objects detected: {len(detections)}")
    for display_name in required_names:
        class_id, model_name = names_by_normalized[display_name]
        print(f"{model_name}: {counts[class_id]}")
    for index, (name, confidence, xyxy) in enumerate(detections, start=1):
        print(
            f"Detection {index}: class={name}, confidence={confidence:.4f}, "
            f"bbox_xyxy={[round(value, 2) for value in xyxy]}"
        )
    print(f"Result image: {output_path}")


if __name__ == "__main__":
    main()
