import unittest

from scripts.create_m1_flip_dataset import (
    EPSILON,
    flip_label_row,
    validate_bbox,
    validate_label,
)


class M1FlipValidationTests(unittest.TestCase):
    def test_bbox_inside_image_is_valid(self) -> None:
        validate_bbox(0, 0.5, 0.5, 0.4, 0.4, "unit test")

    def test_bbox_overshoot_within_epsilon_is_valid(self) -> None:
        validate_bbox(
            0,
            0.5 - 0.99 * EPSILON,
            0.5,
            1.0,
            0.4,
            "unit test",
        )

    def test_bbox_overshoot_beyond_epsilon_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Box extends beyond image"):
            validate_bbox(
                0,
                0.5 - 1.01 * EPSILON,
                0.5,
                1.0,
                0.4,
                "unit test",
            )

    def test_horizontal_flip_preserves_class_and_box_size(self) -> None:
        original = ["1", "0.2", "0.4", "0.3", "0.2"]
        flipped = flip_label_row(original, "unit test")

        self.assertEqual(flipped[0], original[0])
        self.assertAlmostEqual(float(flipped[1]), 1.0 - float(original[1]))
        self.assertEqual(flipped[2:], original[2:])

    def test_previously_failing_source_label_is_valid(self) -> None:
        from pathlib import Path

        label = (
            Path(__file__).resolve().parents[1]
            / "dataset"
            / "helmet_research"
            / "train"
            / "labels"
            / "1007_png.rf.09fc96918d3d6b03a4ad9e0b441d5664.txt"
        )
        rows = validate_label(label)
        flipped = flip_label_row(rows[1], f"{label}:2")

        self.assertEqual(flipped[0], "0")
        self.assertAlmostEqual(float(flipped[1]), 1.0 - 0.021265625)
        self.assertEqual(flipped[2:], rows[1][2:])


if __name__ == "__main__":
    unittest.main()
