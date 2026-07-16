import importlib.util
import struct
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "模型3_200hz原始测试"
MAIN_PATH = TEST_DIR / "main.py"
LINK_PATH = TEST_DIR / "maixcam_link.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTime:
    def __init__(self, timestamp_ms=1234):
        self.timestamp_ms = timestamp_ms

    def ticks_ms(self):
        return self.timestamp_ms


class FakeSensor:
    def __init__(self, data, calibration_exists=True):
        self.data = data
        self.calibration_exists = calibration_exists
        self.loaded_ids = []
        self.read_calls = []

    def calib_gyro_exists(self, save_id):
        return self.calibration_exists

    def load_calib_gyro(self, save_id):
        self.loaded_ids.append(save_id)

    def read_all(self, calib_gryo=True, radian=False):
        self.read_calls.append((calib_gryo, radian))
        return self.data


def vector(x, y, z):
    return types.SimpleNamespace(x=x, y=y, z=z)


class Model3Raw200HzTests(unittest.TestCase):
    def load_main(self):
        link = load_module("raw_200hz_link", LINK_PATH)
        previous = sys.modules.get("maixcam_link")
        sys.modules["maixcam_link"] = link
        try:
            main = load_module("raw_200hz_main", MAIN_PATH)
        finally:
            if previous is None:
                sys.modules.pop("maixcam_link", None)
            else:
                sys.modules["maixcam_link"] = previous
        return main

    def test_raw_source_loads_calibration_and_returns_scaled_six_axis_data(self):
        main = self.load_main()
        data = types.SimpleNamespace(
            acc=vector(0.125, -0.25, 1.0),
            gyro=vector(12.3, -45.6, 0.04),
        )
        sensor = FakeSensor(data)
        source = main.RawImuSource(sensor=sensor, time_module=FakeTime())

        self.assertTrue(source.start())
        self.assertEqual(sensor.loaded_ids, ["model3_gimbal"])
        self.assertEqual(
            source.sample(),
            (1234, 125, -250, 1000, 123, -456, 0, main.FLAG_IMU_VALID),
        )
        self.assertEqual(sensor.read_calls, [(True, False)])

    def test_version_two_frame_contains_timestamp_vision_and_six_axis_data(self):
        link = load_module("raw_200hz_link", LINK_PATH)

        frame = link.build_frame(
            packet_sequence=1,
            vision_frame=2,
            timestamp_ms=3,
            x_error=4,
            y_error=-5,
            acc_x_mg=6,
            acc_y_mg=-7,
            acc_z_mg=8,
            gyro_x_dps_x10=9,
            gyro_y_dps_x10=-10,
            gyro_z_dps_x10=11,
            flags=link.FLAG_TARGET_VALID | link.FLAG_IMU_VALID,
        )

        self.assertEqual(len(frame), 32)
        self.assertEqual(
            struct.unpack("<BBBBHHIhhhhhhhhH", frame[:30]),
            (
                0xA5,
                0x5A,
                0x02,
                32,
                1,
                2,
                3,
                4,
                -5,
                6,
                -7,
                8,
                9,
                -10,
                11,
                link.FLAG_TARGET_VALID | link.FLAG_IMU_VALID,
            ),
        )
        self.assertEqual(
            struct.unpack("<H", frame[30:])[0],
            link.crc16_ccitt_false(frame[:30]),
        )

    def test_test_entrypoint_uses_460800_baud_and_does_not_run_mahony(self):
        text = MAIN_PATH.read_text(encoding="utf-8")

        self.assertIn("UART_BAUDRATE = 460800", text)
        self.assertIn('if __name__ == "__main__":', text)
        self.assertIn("read_all(calib_gryo=True, radian=False)", text)
        self.assertNotIn("MahonyAHRS", text)
        self.assertNotIn("get_angle", text)


if __name__ == "__main__":
    unittest.main()
