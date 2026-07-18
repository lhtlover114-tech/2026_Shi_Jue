import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "模型3" / "find_circle.py"


class Model3HybridDetectionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = TARGET.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.text)

    def test_opencv_is_the_60_fps_primary_path(self):
        required_source = (
            "CAM_W = 640",
            "CAM_H = 480",
            "CAM_FPS = 60",
            "CAM_BUFF_NUM = 3",
            "CAMERA_WARMUP_FRAMES = 20",
            "PROC_W = 320",
            "PROC_H = 240",
            "image.Format.FMT_BGR888",
            "buff_num=CAM_BUFF_NUM",
            "self.cam.skip_frames(CAMERA_WARMUP_FRAMES)",
            "process_points = detect_rectangle(process_frame)",
            "cv2.THRESH_OTSU",
        )
        for snippet in required_source:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, self.text)

        self.assertNotIn("self.cam.constrast(", self.text)
        self.assertNotIn("self.cam.contrast(", self.text)
        self.assertNotIn("cv2.adaptiveThreshold", self.text)

    def test_yolo_only_runs_as_a_rate_limited_roi_rescue(self):
        required_source = (
            "MISS_TRIGGER_FRAMES = 2",
            "YOLO_COOLDOWN_FRAMES = 5",
            "YOLO_CONFIDENCE = 0.25",
            "YOLO_IOU = 0.45",
            "YOLO_ROI_PADDING_RATIO = 0.10",
            "conf_th=YOLO_CONFIDENCE",
            "iou_th=YOLO_IOU",
            "img_ai = img_ai.to_format(self.detector.input_format())",
            "for obj in objs:",
            "rescue_points = detect_rectangle(resized_roi)",
        )
        for snippet in required_source:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, self.text)

        self.assertNotIn("max_idx", self.text)
        self.assertNotIn("img_ai.format()", self.text)

    def test_yolo_failure_cools_down_for_five_complete_frames(self):
        functions = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("update_rescue_schedule", functions)

        namespace = {"MISS_TRIGGER_FRAMES": 2}
        module = ast.Module(
            body=[functions["update_rescue_schedule"]],
            type_ignores=[],
        )
        exec(compile(module, str(TARGET), "exec"), namespace)
        update_schedule = namespace["update_rescue_schedule"]

        miss_streak = 2
        cooldown = 5
        for opencv_valid in (True, False, False, False, False):
            miss_streak, cooldown, should_try = update_schedule(
                miss_streak,
                cooldown,
                opencv_valid,
                True,
            )
            self.assertFalse(should_try)

        self.assertEqual(cooldown, 0)
        miss_streak, cooldown, should_try = update_schedule(
            miss_streak,
            cooldown,
            False,
            True,
        )
        self.assertTrue(should_try)

    def test_runtime_can_disable_an_unhelpful_model(self):
        required_source = (
            "MODEL_EVAL_INTERVAL_MS = 10000",
            "MODEL_DISABLE_INVALID_RATIO = 0.20",
            "MODEL_DISABLE_MIN_ATTEMPTS = 5",
            "MODEL_DISABLE_MAX_RESCUE_RATIO = 0.10",
            "self.yolo_enabled = False",
        )
        for snippet in required_source:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, self.text)

        functions = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("should_disable_yolo", functions)
        namespace = {
            "MODEL_DISABLE_INVALID_RATIO": 0.20,
            "MODEL_DISABLE_MIN_ATTEMPTS": 5,
            "MODEL_DISABLE_MAX_RESCUE_RATIO": 0.10,
        }
        module = ast.Module(
            body=[functions["should_disable_yolo"]],
            type_ignores=[],
        )
        exec(compile(module, str(TARGET), "exec"), namespace)
        should_disable = namespace["should_disable_yolo"]

        self.assertTrue(should_disable(100, 21, 5, 0))
        self.assertFalse(should_disable(100, 20, 5, 0))
        self.assertFalse(should_disable(100, 21, 4, 0))
        self.assertFalse(should_disable(100, 21, 10, 1))

    def test_run_keeps_the_legacy_five_item_interface(self):
        classes = {
            node.name: node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef)
        }
        finder = classes["FindRectCircle"]
        methods = {
            node.name: node
            for node in finder.body
            if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("_to_legacy_center", methods)
        self.assertIn("_to_legacy_error", methods)
        self.assertIn("run", methods)
        self.assertIn("self.last_circle3_points = []", self.text)
        self.assertIn("legacy_center = self._to_legacy_center(center)", self.text)
        self.assertIn(
            "legacy_center[0] - self.center_pos[0]",
            self.text,
        )
        self.assertIn(
            "legacy_center[1] - self.center_pos[1]",
            self.text,
        )

        expected_return = [
            "self.last_center",
            "self.center_pos",
            "self.err_center",
            "self.last_circle3_points",
            "self.updated",
        ]
        list_returns = [
            [ast.unparse(item) for item in node.value.elts]
            for node in ast.walk(methods["run"])
            if isinstance(node, ast.Return) and isinstance(node.value, ast.List)
        ]
        self.assertIn(expected_return, list_returns)

    def test_status_counters_are_reported_once_per_second(self):
        required_source = (
            "STATUS_INTERVAL_MS = 1000",
            '"[vision] fps={:.1f} opencv_valid={:.1f}% "',
            '"miss={} yolo={}/{} enabled={}"',
        )
        for snippet in required_source:
            with self.subTest(snippet=snippet):
                self.assertIn(snippet, self.text)


if __name__ == "__main__":
    unittest.main()
