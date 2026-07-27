import sys
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

from line_metrics import compute_region_raw_errors


class ComputeRegionRawErrorsTests(unittest.TestCase):
    def setUp(self):
        self.strip_y = [30, 44, 58, 72, 87, 101, 115, 130, 144, 158, 172, 187, 201, 215, 230]
        self.center_x = 160.0

    def test_uses_fixed_top_and_bottom_five_strips(self):
        centers = [
            (140, 30, 1),
            (150, 44, 2),
            (170, 72, 4),
            (210, 115, 8),
            (180, 187, 12),
            (190, 215, 14),
            (200, 230, 15),
        ]

        near_error, far_error = compute_region_raw_errors(
            centers, self.center_x, self.strip_y, region_strip_count=5
        )

        expected_far_x = (140 * 1 + 150 * 2 + 170 * 4) / (1 + 2 + 4)
        expected_near_x = (180 * 12 + 190 * 14 + 200 * 15) / (12 + 14 + 15)
        self.assertAlmostEqual(far_error, expected_far_x - self.center_x)
        self.assertAlmostEqual(near_error, expected_near_x - self.center_x)

    def test_positive_is_right_and_negative_is_left(self):
        centers = [(130, 30, 1), (190, 230, 15)]

        near_error, far_error = compute_region_raw_errors(
            centers, self.center_x, self.strip_y, region_strip_count=5
        )

        self.assertEqual(near_error, 30.0)
        self.assertEqual(far_error, -30.0)

    def test_missing_region_returns_none_without_borrowing_middle_points(self):
        centers = [(175, 115, 8)]

        near_error, far_error = compute_region_raw_errors(
            centers, self.center_x, self.strip_y, region_strip_count=5
        )

        self.assertIsNone(near_error)
        self.assertIsNone(far_error)

    def test_region_count_is_clamped_to_available_strips(self):
        strip_y = [20, 40, 60]
        centers = [(150, 20, 1), (160, 40, 2), (170, 60, 3)]

        near_error, far_error = compute_region_raw_errors(
            centers, self.center_x, strip_y, region_strip_count=10
        )

        expected_x = (150 * 1 + 160 * 2 + 170 * 3) / 6
        self.assertAlmostEqual(near_error, expected_x - self.center_x)
        self.assertAlmostEqual(far_error, expected_x - self.center_x)


if __name__ == "__main__":
    unittest.main()
