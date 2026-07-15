import importlib.util
import math
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
ATTITUDE_PATH = ROOT / "模型3" / "imu_attitude.py"


def load_attitude_module():
    spec = importlib.util.spec_from_file_location(
        "model3_imu_attitude", ATTITUDE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTime:
    def __init__(self, *values):
        self.values = list(values)

    def ticks_s(self):
        return self.values.pop(0)


class FakeSensor:
    def __init__(self, data, calibration_exists=True):
        self.data = data
        self.calibration_exists = calibration_exists
        self.loaded_ids = []

    def calib_gyro_exists(self, save_id):
        return self.calibration_exists

    def load_calib_gyro(self, save_id):
        self.loaded_ids.append(save_id)

    def read_all(self, calib_gryo=True, radian=False):
        if not calib_gryo or not radian:
            raise AssertionError("runtime must request calibrated radian data")
        return self.data


def vector(x, y, z):
    return types.SimpleNamespace(x=x, y=y, z=z)


def imu_data():
    return types.SimpleNamespace(
        acc=vector(0.0, 0.0, 1.0),
        gyro=vector(math.radians(-12.5), 0.0, math.radians(30.0)),
        mag=vector(0.0, 0.0, 0.0),
        temp=25.0,
    )


class Model3ImuAttitudeTests(unittest.TestCase):
    def test_missing_calibration_returns_zero_invalid_snapshot(self):
        self.assertTrue(ATTITUDE_PATH.exists())
        module = load_attitude_module()
        sensor = FakeSensor(imu_data(), calibration_exists=False)
        source = module.ImuAttitude(
            sensor=sensor,
            attitude_filter=mock.Mock(),
            time_module=FakeTime(),
        )

        self.assertFalse(source.start())
        self.assertFalse(source.is_calibrated())
        self.assertEqual(source.sample(), (0, 0, 0, 0, 0))
        self.assertEqual(sensor.loaded_ids, [])

    def test_valid_sample_maps_axes_scales_fields_and_sets_flags(self):
        self.assertTrue(ATTITUDE_PATH.exists())
        module = load_attitude_module()
        sensor = FakeSensor(imu_data())
        attitude_filter = mock.Mock()
        attitude_filter.get_angle.return_value = vector(-10.0, 2.0, 45.0)
        source = module.ImuAttitude(
            sensor=sensor,
            attitude_filter=attitude_filter,
            time_module=FakeTime(1.000, 1.005),
            settle_samples=1,
        )

        self.assertTrue(source.start())
        self.assertEqual(sensor.loaded_ids, ["model3_gimbal"])
        self.assertEqual(
            source.sample(),
            (
                300,
                -125,
                4500,
                -1000,
                module.FLAG_IMU_VALID | module.FLAG_ATTITUDE_VALID,
            ),
        )
        attitude_filter.get_angle.assert_called_once()
        self.assertAlmostEqual(attitude_filter.get_angle.call_args.args[3], 0.005)
        self.assertFalse(attitude_filter.get_angle.call_args.kwargs["radian"])

    def test_large_dt_keeps_rates_but_clears_attitude(self):
        self.assertTrue(ATTITUDE_PATH.exists())
        module = load_attitude_module()
        attitude_filter = mock.Mock()
        source = module.ImuAttitude(
            sensor=FakeSensor(imu_data()),
            attitude_filter=attitude_filter,
            time_module=FakeTime(1.000, 1.100),
            settle_samples=1,
        )

        self.assertTrue(source.start())
        self.assertEqual(
            source.sample(),
            (300, -125, 0, 0, module.FLAG_IMU_VALID),
        )
        attitude_filter.get_angle.assert_not_called()

    def test_attitude_valid_waits_for_settle_sample_count(self):
        self.assertTrue(ATTITUDE_PATH.exists())
        module = load_attitude_module()
        attitude_filter = mock.Mock()
        attitude_filter.get_angle.return_value = vector(1.0, 2.0, 3.0)
        source = module.ImuAttitude(
            sensor=FakeSensor(imu_data()),
            attitude_filter=attitude_filter,
            time_module=FakeTime(1.000, 1.005, 1.010),
            settle_samples=2,
        )

        self.assertTrue(source.start())
        self.assertEqual(source.sample()[4], module.FLAG_IMU_VALID)
        self.assertEqual(
            source.sample()[4],
            module.FLAG_IMU_VALID | module.FLAG_ATTITUDE_VALID,
        )


if __name__ == "__main__":
    unittest.main()
