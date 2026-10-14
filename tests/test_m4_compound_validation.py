import unittest

from scripts.create_m4_compound_dataset import transform_box, validate_bbox


class M4CompoundValidationTests(unittest.TestCase):
    def test_combined_flip_and_zero_degree_rotation_mirrors_box_center(self) -> None:
        transformed = transform_box(
            ["1", "0.2", "0.4", "0.3", "0.2"],
            100,
            100,
            0.0,
            "unit test",
        )

        self.assertIsNotNone(transformed)
        assert transformed is not None
        self.assertEqual(transformed[0], "1")
        self.assertAlmostEqual(float(transformed[1]), 0.8)
        self.assertAlmostEqual(float(transformed[2]), 0.4)
        self.assertAlmostEqual(float(transformed[3]), 0.3)
        self.assertAlmostEqual(float(transformed[4]), 0.2)

    def test_rotation_keeps_transformed_box_inside_image(self) -> None:
        transformed = transform_box(
            ["0", "0.35", "0.5", "0.2", "0.1"],
            200,
            100,
            15.0,
            "unit test",
        )

        self.assertIsNotNone(transformed)
        assert transformed is not None
        validate_bbox(
            int(transformed[0]),
            *map(float, transformed[1:]),
            "unit test",
        )

    def test_invalid_class_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid class id"):
            transform_box(
                ["2", "0.5", "0.5", "0.2", "0.2"],
                100,
                100,
                0.0,
                "unit test",
            )

    def test_bbox_outside_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "extends beyond image"):
            validate_bbox(0, 0.05, 0.5, 0.2, 0.2, "unit test")


if __name__ == "__main__":
    unittest.main()
