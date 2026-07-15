import ast
import importlib.util
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL3_DIR = ROOT / "模型3"
LINK_PATH = MODEL3_DIR / "maixcam_link.py"
MAIN_PATH = MODEL3_DIR / "main.py"


def load_link_module():
    spec = importlib.util.spec_from_file_location("model3_maixcam_link", LINK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model3MaixCamLinkTests(unittest.TestCase):
    def test_crc16_ccitt_false_standard_vector(self):
        link = load_link_module()
        self.assertEqual(link.crc16_ccitt_false(b"123456789"), 0x29B1)

    def test_frame_matches_mspm0_layout_and_crc(self):
        link = load_link_module()

        frame = link.build_frame(
            packet_sequence=0x1234,
            vision_frame=0x5678,
            timestamp_ms=0x9ABCDEF0,
            x_error=321,
            y_error=-123,
            yaw_rate_x10=11,
            pitch_rate_x10=-22,
            yaw_angle_x100=333,
            pitch_angle_x100=-444,
            flags=link.FLAG_TARGET_VALID,
        )

        self.assertEqual(len(frame), 28)
        fields = struct.unpack("<BBBBHHIhhhhhhH", frame[:26])
        self.assertEqual(
            fields,
            (
                0xA5,
                0x5A,
                0x01,
                28,
                0x1234,
                0x5678,
                0x9ABCDEF0,
                321,
                -123,
                11,
                -22,
                333,
                -444,
                link.FLAG_TARGET_VALID,
            ),
        )
        self.assertEqual(
            struct.unpack("<H", frame[26:])[0],
            link.crc16_ccitt_false(frame[:26]),
        )

    def test_frame_wraps_unsigned_and_saturates_signed_fields(self):
        link = load_link_module()

        frame = link.build_frame(
            packet_sequence=0x10001,
            vision_frame=-1,
            timestamp_ms=0x100000002,
            x_error=40000,
            y_error=-40000,
            yaw_rate_x10=0,
            pitch_rate_x10=0,
            yaw_angle_x100=0,
            pitch_angle_x100=0,
            flags=0x10001,
        )

        fields = struct.unpack("<BBBBHHIhhhhhhH", frame[:26])
        self.assertEqual(fields[4:9], (1, 0xFFFF, 2, 32767, -32768))
        self.assertEqual(fields[-1], 1)

    def test_target_is_valid_before_timeout_and_cleared_at_timeout(self):
        link = load_link_module()
        snapshot = (7, 12, -34, 1000)

        self.assertEqual(
            link.resolve_target_snapshot(snapshot, 1199, 200),
            (7, 12, -34, link.FLAG_TARGET_VALID),
        )
        self.assertEqual(
            link.resolve_target_snapshot(snapshot, 1200, 200),
            (7, 0, 0, 0),
        )
        self.assertEqual(
            link.resolve_target_snapshot(None, 1200, 200),
            (0, 0, 0, 0),
        )

    def test_main_uses_find_circle_and_publishes_only_updated_results(self):
        text = MAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(text)

        self.assertIn("from find_circle import FindRectCircle", text)
        self.assertIn("from maixcam_link import MaixCamLink", text)
        self.assertNotIn("nn.YOLO11", text)
        self.assertNotIn("detector.detect", text)
        self.assertIn("finder.debug_draw_circle = False", text)

        main_functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        ]
        self.assertEqual(len(main_functions), 1)
        main_function = main_functions[0]

        publish_calls = [
            node
            for node in ast.walk(main_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "publish_target"
        ]
        self.assertEqual(len(publish_calls), 1)

        guarded_publish = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "updated"
            and any(call is publish_calls[0] for call in ast.walk(node))
            for node in ast.walk(main_function)
        )
        self.assertTrue(guarded_publish)


if __name__ == "__main__":
    unittest.main()
