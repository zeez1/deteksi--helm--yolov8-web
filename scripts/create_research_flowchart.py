"""Create a publication-ready research flowchart from verified project stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
OUTPUT_PNG = REPORTS_DIR / "diagram_alur_penelitian.png"
OUTPUT_PDF = REPORTS_DIR / "diagram_alur_penelitian.pdf"

CANVAS_WIDTH = 12.0
CANVAS_HEIGHT = 16.0
NAVY = "#18324B"
BLUE = "#DDEBF7"
PALE_BLUE = "#EFF5FA"
GREEN = "#E6F2EA"
PALE_GREEN = "#F2F7F3"
GOLD = "#FFF1D6"
GREY = "#F1F3F5"
TEXT = "#1F2933"
LINE = "#526575"

EXPERIMENTS = (
    ("M0", "Baseline — tanpa augmentasi"),
    ("M1", "Horizontal flip offline pada data train"),
    ("M2", "Rotasi offline ±15° pada data train"),
    ("M3", "Brightness 0,8 + contrast 1,2 offline pada data train"),
    (
        "M4",
        "Compound offline pada data train: flip + rotasi ±15° + "
        "brightness 0,8 + contrast 1,2",
    ),
    (
        "M5",
        "Controlled online pada data train: flip, rotasi ±5°, scale 0,1, "
        "translate 0,05, brightness/contrast (p=0,2)",
    ),
)


def add_box(
    axis: plt.Axes,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = BLUE,
    edgecolor: str = NAVY,
    fontsize: float = 10.5,
    fontweight: str = "normal",
    radius: float = 0.012,
    textcolor: str = TEXT,
) -> None:
    box = FancyBboxPatch(
        (center_x - width / 2, center_y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
        zorder=2,
    )
    axis.add_patch(box)
    axis.text(
        center_x,
        center_y,
        text,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=fontsize,
        fontweight=fontweight,
        wrap=True,
        zorder=3,
    )


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    arrowstyle: str = "-|>",
    linewidth: float = 1.25,
    color: str = LINE,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": arrowstyle,
            "color": color,
            "linewidth": linewidth,
            "shrinkA": 0,
            "shrinkB": 0,
            "mutation_scale": 12,
        },
        zorder=1,
    )


def create_diagram() -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        figure, axis = plt.subplots(figsize=(CANVAS_WIDTH, CANVAS_HEIGHT))
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        figure.subplots_adjust(left=0.065, right=0.935, top=0.95, bottom=0.045)

        axis.text(
            0.5,
            0.976,
            "Diagram Alur Penelitian",
            ha="center",
            va="center",
            fontsize=19,
            fontweight="bold",
            color=NAVY,
        )
        axis.text(
            0.5,
            0.954,
            "Analisis augmentasi data pada YOLOv8 untuk deteksi penggunaan helm",
            ha="center",
            va="center",
            fontsize=10.5,
            color=TEXT,
        )

        node_x = 0.5
        node_width = 0.78
        node_height = 0.052
        common_nodes = (
            (
                0.905,
                "PERSIAPAN DAN AUDIT DATASET",
                "908 gambar + 908 label; kelas helmet dan no-helmet; "
                "audit anotasi/gambar tanpa temuan fatal",
                BLUE,
            ),
            (
                0.827,
                "PEMBAGIAN DATASET DAN VALIDASI SPLIT",
                "Train 635 | Validation 181 | Test 92; "
                "0 mismatch image-label dan 0 duplikasi lintas split",
                PALE_BLUE,
            ),
            (
                0.749,
                "MODEL DAN PROTOKOL EKSPERIMEN",
                "YOLOv8n pretrained; 50 epoch; imgsz 640; "
                "validasi selama training; test tidak digunakan untuk training",
                BLUE,
            ),
        )

        for center_y, heading, detail, color in common_nodes:
            add_box(
                axis,
                node_x,
                center_y,
                node_width,
                node_height,
                f"{heading}\n{detail}",
                facecolor=color,
                fontsize=9.5,
                fontweight="normal",
            )

        for upper_y, lower_y in ((0.879, 0.853), (0.801, 0.775)):
            add_arrow(axis, (node_x, upper_y), (node_x, lower_y))

        axis.text(
            0.5,
            0.697,
            "ENAM EKSPERIMEN DENGAN SPLIT EVALUASI YANG SAMA",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=NAVY,
        )

        input_bus_x = 0.075
        output_bus_x = 0.925
        branch_top_y = 0.68
        experiment_centers = (0.635, 0.581, 0.527, 0.473, 0.419, 0.365)
        branch_width = 0.74
        branch_height = 0.041

        # A shared input bus branches to all six experiment configurations.
        axis.plot(
            [node_x, node_x, input_bus_x],
            [0.723, 0.715, 0.715],
            color=LINE,
            linewidth=1.25,
            zorder=1,
        )
        add_arrow(axis, (input_bus_x, 0.715), (input_bus_x, branch_top_y))
        axis.plot(
            [input_bus_x, input_bus_x],
            [branch_top_y, experiment_centers[-1]],
            color=LINE,
            linewidth=1.25,
            zorder=1,
        )

        # Each experiment returns to a common output bus before evaluation.
        axis.plot(
            [output_bus_x, output_bus_x],
            [experiment_centers[0], 0.307],
            color=LINE,
            linewidth=1.25,
            zorder=1,
        )

        for center_y, (model, description) in zip(
            experiment_centers, EXPERIMENTS, strict=True
        ):
            add_box(
                axis,
                0.5,
                center_y,
                branch_width,
                branch_height,
                f"{model}  |  {description}",
                facecolor=PALE_GREEN if model in {"M0", "M5"} else GREEN,
                fontsize=9.3,
            )
            add_arrow(axis, (input_bus_x, center_y), (0.13, center_y))
            add_arrow(axis, (0.87, center_y), (output_bus_x, center_y))

        evaluation_y = 0.266
        add_arrow(axis, (output_bus_x, 0.307), (output_bus_x, evaluation_y))
        axis.plot(
            [output_bus_x, node_x],
            [evaluation_y, evaluation_y],
            color=LINE,
            linewidth=1.25,
            zorder=1,
        )
        add_arrow(axis, (node_x, evaluation_y), (node_x, evaluation_y - 0.009))
        add_box(
            axis,
            node_x,
            0.232,
            0.82,
            0.057,
            "EVALUASI PADA TEST SET (92 GAMBAR)\n"
            "Precision  |  Recall  |  mAP50  |  mAP50-95",
            facecolor=GOLD,
            fontsize=10,
            fontweight="bold",
        )

        add_arrow(axis, (node_x, 0.203), (node_x, 0.183))
        add_box(
            axis,
            node_x,
            0.153,
            0.82,
            0.055,
            "PERBANDINGAN HASIL M0–M5 DAN PEMILIHAN KONFIGURASI\n"
            "M5 tercatat terbaik pada mAP50 dan mAP50-95",
            facecolor=GOLD,
            fontsize=9.7,
            fontweight="bold",
        )

        add_arrow(axis, (node_x, 0.125), (node_x, 0.102))
        add_box(
            axis,
            node_x,
            0.072,
            0.82,
            0.055,
            "IMPLEMENTASI DETEKSI BERBASIS WEB DENGAN MODEL M5\n"
            "Aplikasi Streamlit tersedia untuk input foto dan video",
            facecolor=GREY,
            fontsize=9.7,
            fontweight="bold",
        )
        axis.text(
            0.5,
            0.025,
            "Catatan: implementasi aplikasi tersedia; uji formal dan deployment publik "
            "tidak terverifikasi dalam artefak yang diperiksa.",
            ha="center",
            va="center",
            fontsize=8.2,
            color=TEXT,
        )

        figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
        figure.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight")
        plt.close(figure)

    return OUTPUT_PNG, OUTPUT_PDF


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Buat diagram alur penelitian M0–M5 dalam PNG dan PDF."
    )
    parser.parse_args()

    png_path, pdf_path = create_diagram()
    print(f"PNG berhasil dibuat: {png_path}")
    print(f"PDF berhasil dibuat: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
