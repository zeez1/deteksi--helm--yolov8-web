"""Create a publication-ready comparison chart for the M0-M5 test results."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
COMPARISON_CSV = REPORTS_DIR / "M0_M1_M2_M3_M4_M5_comparison.csv"
EXPECTED_MODELS = ("M0", "M1", "M2", "M3", "M4", "M5")
METRICS = (
    ("Precision", "Precision", "precision"),
    ("Recall", "Recall", "recall"),
    ("mAP50", "mAP50", "mAP50"),
    ("mAP50-95", "mAP50-95", "mAP50_95"),
)
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")


def read_comparison_data() -> list[dict[str, Any]]:
    if not COMPARISON_CSV.is_file():
        raise FileNotFoundError(f"Laporan perbandingan tidak ditemukan: {COMPARISON_CSV}")

    with COMPARISON_CSV.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "Model",
            "Augmentation",
            *(column for _, column, _ in METRICS),
        }
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "Kolom wajib tidak ditemukan di laporan perbandingan: "
                + ", ".join(sorted(missing_columns))
            )
        rows = list(reader)

    actual_models = tuple(row.get("Model", "").strip() for row in rows)
    if actual_models != EXPECTED_MODELS:
        raise ValueError(
            "Model dalam laporan harus tepat dan berurutan M0–M5; "
            f"ditemukan: {actual_models}"
        )

    for row in rows:
        if not row.get("Augmentation", "").strip():
            raise ValueError(f"Deskripsi augmentasi kosong untuk {row['Model']}.")
        for _, column, _ in METRICS:
            raw_value = row[column].strip()
            try:
                value = float(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"Nilai {column} untuk {row['Model']} bukan angka: {raw_value!r}"
                ) from error
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(
                    f"Nilai {column} untuk {row['Model']} harus berada pada 0–1: "
                    f"{raw_value!r}"
                )
    return rows


def verify_test_reports(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        model = row["Model"]
        report_path = REPORTS_DIR / f"{model}_test_results.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Laporan hasil test tidak ditemukan: {report_path}")
        with report_path.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)

        if not isinstance(report, dict):
            raise ValueError(f"Format {report_path.name} harus berupa objek JSON.")
        if report.get("split") != "test":
            raise ValueError(
                f"{report_path.name} bukan hasil evaluasi split test."
            )
        if not str(report.get("experiment", "")).startswith(f"{model}_"):
            raise ValueError(
                f"Nama eksperimen pada {report_path.name} tidak sesuai dengan {model}."
            )
        if not isinstance(report.get("metrics"), dict):
            raise ValueError(f"Bagian metrics tidak tersedia di {report_path.name}.")

        for _, csv_column, json_key in METRICS:
            raw_csv_value = row[csv_column].strip()
            csv_value = Decimal(raw_csv_value)
            csv_decimal_places = (
                len(raw_csv_value.partition(".")[2])
                if "." in raw_csv_value
                else 0
            )
            tolerance = (
                Decimal(5).scaleb(-(csv_decimal_places + 1))
                + Decimal("1e-15")
            )
            json_value = report["metrics"].get(json_key)
            if (
                isinstance(json_value, bool)
                or not isinstance(json_value, (int, float))
                or not math.isfinite(float(json_value))
            ):
                raise ValueError(
                    f"Metrik {json_key} tidak valid di {report_path.name}."
                )
            report_value = Decimal(str(json_value))
            if abs(csv_value - report_value) > tolerance:
                raise ValueError(
                    f"Perbedaan data untuk {model} {csv_column}: CSV={csv_value}, "
                    f"{report_path.name}={report_value}; grafik tidak dibuat."
                )


def create_chart(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    output_png = REPORTS_DIR / "grafik_perbandingan_m0_m5.png"
    output_pdf = REPORTS_DIR / "grafik_perbandingan_m0_m5.pdf"

    values = np.array(
        [
            [float(row[column]) for row in rows]
            for _, column, _ in METRICS
        ],
        dtype=float,
    )
    group_positions = np.arange(len(rows))
    bar_width = 0.19
    offsets = (np.arange(len(METRICS)) - (len(METRICS) - 1) / 2) * bar_width
    labels = []
    for row in rows:
        description = textwrap.fill(
            row["Augmentation"].strip(),
            width=24,
            break_long_words=False,
            break_on_hyphens=False,
        )
        labels.append(f"{row['Model']}\n{description}")
    max_label_lines = max(label.count("\n") + 1 for label in labels)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        figure, axis = plt.subplots(figsize=(14, 8))
        figure.subplots_adjust(
            left=0.08,
            right=0.99,
            top=0.82,
            bottom=min(0.42, 0.25 + max(0, max_label_lines - 2) * 0.045),
        )

        for metric_index, (metric_label, _, _) in enumerate(METRICS):
            positions = group_positions + offsets[metric_index]
            bars = axis.bar(
                positions,
                values[metric_index],
                width=bar_width * 0.9,
                label=metric_label,
                color=COLORS[metric_index],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.3f}" for value in values[metric_index]],
                padding=2,
                fontsize=7.5,
                color="#222222",
            )

        axis.set_title(
            "Perbandingan Performa YOLOv8 pada Eksperimen M0–M5",
            fontweight="bold",
            pad=18,
        )
        axis.set_xlabel("Eksperimen", labelpad=12)
        axis.set_ylabel("Nilai metrik")
        axis.set_xticks(group_positions, labels)
        axis.set_ylim(0, 1)
        axis.set_yticks(np.arange(0, 1.01, 0.1))
        axis.set_xlim(-0.6, len(rows) - 0.4)
        axis.yaxis.grid(True, color="#D9DEE5", linewidth=0.7, zorder=0)
        axis.xaxis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#AAB2BD")
        axis.tick_params(axis="both", length=0, pad=6)
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.17),
            ncol=len(METRICS),
            frameon=False,
            handlelength=1.4,
            columnspacing=2.0,
        )

        figure.savefig(output_png, dpi=300, bbox_inches="tight")
        figure.savefig(output_pdf, format="pdf", bbox_inches="tight")
        plt.close(figure)

    return output_png, output_pdf


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Buat grafik batang perbandingan metrik test M0–M5 dari laporan "
            "proyek dan verifikasi nilainya terhadap laporan JSON."
        )
    )
    parser.parse_args()

    try:
        rows = read_comparison_data()
        verify_test_reports(rows)
        output_png, output_pdf = create_chart(rows)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        parser.error(str(error))

    print(f"PNG berhasil dibuat: {output_png}")
    print(f"PDF berhasil dibuat: {output_pdf}")
    print("CSV data verified against M0-M5 test JSON reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
