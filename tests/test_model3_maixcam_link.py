import ast
import importlib.util
import inspect
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


class Model3VisionOnlyLinkTests(unittest.TestCase):
    def test_crc16_ccitt_false_standard_vector(self):
        link = load_link_module()
        self.assertEqual(link.crc16_ccitt_false(b"123456789"), 0x29B1)

    def test_v2_frame_contains_six_axis_placeholders_and_crc(self):
        link = load_link_module()
        frame = link.build_frame(
            packet_sequence=0x1234,
            vision_frame=0x5678,
            timestamp_ms=0x9ABCDEF0,
            x_error=321,
            y_error=-123,
            flags=link.FLAG_TARGET_VALID,
        )

        self.assertEqual(len(frame), 32)
        self.assertEqual(
            struct.unpack("<BBBBHHIhhhhhhhhH", frame[:30]),
            (
                0xA5,
                0x5A,
                0x02,
                32,
                0x1234,
                0x5678,
                0x9ABCDEF0,
                321,
                -123,
                0,
                0,
                0,
                0,
                0,
                0,
                link.FLAG_TARGET_VALID,
            ),
        )
        self.assertEqual(
            struct.unpack("<H", frame[30:])[0],
            link.crc16_ccitt_false(frame[:30]),
        )

    def test_build_frame_keeps_future_six_axis_fields_available(self):
        link = load_link_module()
        frame = link.build_frame(
            packet_sequence=1,
            vision_frame=2,
            timestamp_ms=3,
            x_error=4,
            y_error=5,
            acc_x_mg=6,
            acc_y_mg=7,
            acc_z_mg=8,
            gyro_x_dps_x10=9,
            gyro_y_dps_x10=10,
            gyro_z_dps_x10=11,
            flags=link.FLAG_IMU_VALID,
        )

        fields = struct.unpack("<BBBBHHIhhhhhhhhH", frame[:30])
        self.assertEqual(fields[9:15], (6, 7, 8, 9, 10, 11))
        self.assertEqual(fields[15], link.FLAG_IMU_VALID)

    def test_target_keeps_last_value_without_timeout(self):
        link = load_link_module()
        snapshot = (7, 12, -34)

        self.assertEqual(
            link.resolve_target_snapshot(snapshot),
            (7, 12, -34, link.FLAG_TARGET_VALID),
        )
        self.assertEqual(link.resolve_target_snapshot(None), (0, 0, 0, 0))

    def test_publish_target_replaces_held_value(self):
        link = load_link_module()
        instance = link.MaixCamLink()
        instance._started = True

        instance.publish_target(12, -34)
        self.assertEqual(instance._target_snapshot, (1, 12, -34))

        instance.publish_target(56, -78)
        self.assertEqual(instance._target_snapshot, (2, 56, -78))

    def test_link_has_no_runtime_imu_source(self):
        link = load_link_module()

        self.assertNotIn(
            "attitude_source",
            inspect.signature(link.MaixCamLink).parameters,
        )
        self.assertNotIn("imu_source", inspect.signature(link.MaixCamLink).parameters)
        self.assertNotIn(
            "target_timeout_ms",
            inspect.signature(link.MaixCamLink).parameters,
        )
        self.assertEqual(link.MaixCamLink().get_stats(), (0, 0, 0))

    def test_worker_sends_latest_target_with_zero_imu_fields(self):
        link = load_link_module()
        instance = link.MaixCamLink()

        class FakeApp:
            def __init__(self):
                self.calls = 0

            def need_exit(self):
                self.calls += 1
                return self.calls > 1

        class FakeTime:
            @staticmethod
            def ticks_us():
                return 0

            @staticmethod
            def ticks_ms():
                return 1000

            @staticmethod
            def sleep_us(_):
                return None

        class FakeSerial:
            def __init__(self):
                self.frames = []

            def write(self, frame):
                self.frames.append(frame)
                return len(frame)

        serial = FakeSerial()
        instance._app = FakeApp()
        instance._time = FakeTime()
        instance._serial = serial
        instance._target_snapshot = (9, 10, -11)
        instance._tx_worker(None)

        fields = struct.unpack("<BBBBHHIhhhhhhhhH", serial.frames[0][:30])
        self.assertEqual(fields[5:9], (9, 1000, 10, -11))
        self.assertEqual(fields[9:15], (0, 0, 0, 0, 0, 0))
        self.assertEqual(fields[15], link.FLAG_TARGET_VALID)

    def test_main_runs_find_circle_and_does_not_start_imu(self):
        text = MAIN_PATH.read_text(encoding="utf-8")
        tree = ast.parse(text)

        self.assertIn("from find_circle import FindRectCircle", text)
        self.assertIn("from maixcam_link import MaixCamLink", text)
        self.assertNotIn("from imu_attitude import", text)
        self.assertNotIn("attitude_source=", text)
        self.assertNotIn("imu_source=", text)
        self.assertNotIn("target_timeout_ms=", text)
        self.assertIn("UART_BAUDRATE = 460800", text)
        self.assertIn("finder.debug_draw_rect = True", text)
        self.assertIn("link.publish_target(x_error, y_error)", text)
        self.assertNotIn("link.get_imu_stats()", text)

        main_function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        publish_calls = [
            node for node in ast.walk(main_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "publish_target"
        ]
        self.assertEqual(len(publish_calls), 1)


if __name__ == "__main__":
    unittest.main()
