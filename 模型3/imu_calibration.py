# -*- coding: utf-8 -*-
"""MaixCAM2 板载陀螺仪独立校准入口。"""


CALIBRATION_ENABLE = False
CALIBRATION_TIME_MS = 10000
CALIBRATION_SAVE_ID = "model3_gimbal"
CALIBRATION_PREPARE_MS = 3000


def _create_sensor(imu):
    return imu.IMU(
        "default",
        mode=imu.Mode.DUAL,
        acc_scale=imu.AccScale.ACC_SCALE_4G,
        acc_odr=imu.AccOdr.ACC_ODR_416,
        gyro_scale=imu.GyroScale.GYRO_SCALE_1000DPS,
        gyro_odr=imu.GyroOdr.GYRO_ODR_416,
    )


def main():
    if not CALIBRATION_ENABLE:
        print("[imu-cal] disabled; no calibration data was changed")
        return 0

    from maix import time
    from maix.ext_dev import imu

    sensor = _create_sensor(imu)
    print("[imu-cal] keep MaixCAM2, chassis and gimbal still")
    time.sleep_ms(CALIBRATION_PREPARE_MS)
    bias = sensor.calib_gyro(
        CALIBRATION_TIME_MS,
        save_id=CALIBRATION_SAVE_ID,
    )
    print(
        "[imu-cal] saved bias: {:.6f}, {:.6f}, {:.6f}".format(
            bias.x,
            bias.y,
            bias.z,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
