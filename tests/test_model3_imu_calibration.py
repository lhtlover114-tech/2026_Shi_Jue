import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = ROOT / "模型3" / "imu_calibration.py"


def load_calibration_module():
    spec = importlib.util.spec_from_file_location(
        "model3_imu_calibration", CALIBRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Model3ImuCalibrationTests(unittest.TestCase):
    def test_calibration_is_disabled_by_default_and_does_not_import_maix(self):
        self.assertTrue(CALIBRATION_PATH.exists())
        with mock.patch.dict(sys.modules, {"maix": None}):
            module = load_calibration_module()
            self.assertFalse(module.CALIBRATION_ENABLE)
            self.assertEqual(module.main(), 0)

    def test_enabled_calibration_uses_fixed_profile_and_saves_bias(self):
        self.assertTrue(CALIBRATION_PATH.exists())
        sensor = mock.Mock()
        sensor.calib_gyro.return_value = types.SimpleNamespace(
            x=0.1, y=-0.2, z=0.3
        )

        class Mode:
            DUAL = "dual"

        class AccScale:
            ACC_SCALE_4G = "acc-4g"

        class AccOdr:
            ACC_ODR_416 = "acc-416"

        class GyroScale:
            GYRO_SCALE_1000DPS = "gyro-1000dps"

        class GyroOdr:
            GYRO_ODR_416 = "gyro-416"

        imu_module = types.ModuleType("maix.ext_dev.imu")
        imu_module.Mode = Mode
        imu_module.AccScale = AccScale
        imu_module.AccOdr = AccOdr
        imu_module.GyroScale = GyroScale
        imu_module.GyroOdr = GyroOdr
        imu_module.IMU = mock.Mock(return_value=sensor)

        time_module = types.SimpleNamespace(sleep_ms=mock.Mock())
        maix_module = types.ModuleType("maix")
        maix_module.time = time_module
        ext_dev_module = types.ModuleType("maix.ext_dev")
        ext_dev_module.imu = imu_module

        fake_modules = {
            "maix": maix_module,
            "maix.ext_dev": ext_dev_module,
            "maix.ext_dev.imu": imu_module,
        }
        with mock.patch.dict(sys.modules, fake_modules):
            module = load_calibration_module()
            module.CALIBRATION_ENABLE = True
            self.assertEqual(module.main(), 0)

        imu_module.IMU.assert_called_once_with(
            "default",
            mode="dual",
            acc_scale="acc-4g",
            acc_odr="acc-416",
            gyro_scale="gyro-1000dps",
            gyro_odr="gyro-416",
        )
        time_module.sleep_ms.assert_called_once_with(3000)
        sensor.calib_gyro.assert_called_once_with(
            10000, save_id="model3_gimbal"
        )


if __name__ == "__main__":
    unittest.main()
