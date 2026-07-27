import importlib
import sys
import types
import unittest
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEST_ROOT))

maix = types.ModuleType("maix")
maix.camera = types.SimpleNamespace()
maix.display = types.SimpleNamespace()
maix.image = types.SimpleNamespace()
maix.app = types.SimpleNamespace()
maix.err = types.SimpleNamespace()
maix.pinmap = types.SimpleNamespace()
maix.thread = types.SimpleNamespace()
maix.uart = types.SimpleNamespace()
maix.time = types.SimpleNamespace(fps=lambda: 50.0)
sys.modules["maix"] = maix
sys.modules.pop("find_line", None)
find_line = importlib.import_module("find_line")


class LineFollowerProcessTests(unittest.TestCase):
    def make_follower(self, centers):
        follower = find_line.LineFollower.__new__(find_line.LineFollower)
        follower._center_x = 160.0
        follower._strip_y = [30, 44, 58, 72, 87, 101, 115, 130, 144, 158, 172, 187, 201, 215, 230]
        follower.NUM_STRIPS = 15
        follower.REGION_STRIPS = 5
        follower.SMOOTH_FACTOR = 0.5
        follower.last_error = 0.0
        follower.last_near_error = 0.0
        follower.last_far_error = 0.0
        follower.last_center_x = 160.0
        follower.last_points = []
        follower.fps = 0.0
        follower.DEBUG = False
        follower.DEBUG_PRINT = False
        follower.cam = types.SimpleNamespace(read=lambda: object())
        follower.disp = types.SimpleNamespace(show=lambda image: None)
        follower._detect_all_strips = lambda image: centers
        return follower

    def test_process_returns_independently_filtered_near_and_far_errors(self):
        centers = [
            (140, 30, 1, 0, 0, 1, 1),
            (140, 44, 2, 0, 0, 1, 1),
            (180, 215, 14, 0, 0, 1, 1),
            (180, 230, 15, 0, 0, 1, 1),
        ]
        follower = self.make_follower(centers)

        result = follower.process()

        self.assertEqual(result["near_error"], 10.0)
        self.assertEqual(result["far_error"], -10.0)
        self.assertIn("error", result)
        self.assertEqual(result["fps"], 50.0)

    def test_missing_near_region_keeps_last_near_error(self):
        centers = [(140, 30, 1, 0, 0, 1, 1)]
        follower = self.make_follower(centers)
        follower.last_near_error = 7.5

        result = follower.process()

        self.assertEqual(result["near_error"], 7.5)
        self.assertEqual(result["far_error"], -10.0)


if __name__ == "__main__":
    unittest.main()
