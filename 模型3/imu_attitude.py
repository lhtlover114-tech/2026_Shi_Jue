# -*- coding: utf-8 -*-
"""模型3运行期板载 IMU 读取与姿态解算。"""


FLAG_IMU_VALID = 1 << 1
FLAG_ATTITUDE_VALID = 1 << 2

CALIBRATION_SAVE_ID = "model3_gimbal"
MAX_DT_S = 0.050
ATTITUDE_SETTLE_SAMPLES = 20
RAD_TO_DEG = 57.29577951308232

# MaixCAM2 板载坐标系默认映射，上板后只需在这里调整轴和符号。
YAW_AXIS = "z"
YAW_SIGN = 1.0
PITCH_AXIS = "x"
PITCH_SIGN = 1.0


class ImuAttitude:
    """加载既有校准，并按调用节拍产生一次 IMU/姿态快照。"""

    def __init__(
        self,
        sensor=None,
        attitude_filter=None,
        time_module=None,
        settle_samples=ATTITUDE_SETTLE_SAMPLES,
    ):
        self._sensor = sensor
        self._filter = attitude_filter
        self._time = time_module
        self._settle_samples = max(1, int(settle_samples))
        self._sample_count = 0
        self._last_time_s = None
        self._calibrated = False

    def _create_hardware(self):
        from maix import ahrs, time
        from maix.ext_dev import imu

        self._time = time
        self._sensor = imu.IMU(
            "default",
            mode=imu.Mode.DUAL,
            acc_scale=imu.AccScale.ACC_SCALE_4G,
            acc_odr=imu.AccOdr.ACC_ODR_416,
            gyro_scale=imu.GyroScale.GYRO_SCALE_1000DPS,
            gyro_odr=imu.GyroOdr.GYRO_ODR_416,
        )
        self._filter = ahrs.MahonyAHRS(2.0, 0.01)

    def start(self):
        """初始化硬件并加载独立校准模块保存的零偏。"""
        if self._sensor is None:
            self._create_hardware()

        if not self._sensor.calib_gyro_exists(CALIBRATION_SAVE_ID):
            self._calibrated = False
            return False

        self._sensor.load_calib_gyro(CALIBRATION_SAVE_ID)
        self._calibrated = True
        self._sample_count = 0
        self._last_time_s = self._time.ticks_s()
        return True

    def sample(self):
        """返回 Yaw/Pitch 角速度、角度和有效位的定点数快照。"""
        if not self._calibrated:
            return (0, 0, 0, 0, 0)

        data = self._sensor.read_all(calib_gryo=True, radian=True)
        now_s = self._time.ticks_s()
        dt = now_s - self._last_time_s
        self._last_time_s = now_s

        yaw_rate_dps = (
            getattr(data.gyro, YAW_AXIS) * RAD_TO_DEG * YAW_SIGN
        )
        pitch_rate_dps = (
            getattr(data.gyro, PITCH_AXIS) * RAD_TO_DEG * PITCH_SIGN
        )
        yaw_rate_x10 = int(round(yaw_rate_dps * 10.0))
        pitch_rate_x10 = int(round(pitch_rate_dps * 10.0))

        if dt <= 0.0 or dt > MAX_DT_S:
            return (
                yaw_rate_x10,
                pitch_rate_x10,
                0,
                0,
                FLAG_IMU_VALID,
            )

        angle = self._filter.get_angle(
            data.acc,
            data.gyro,
            data.mag,
            dt,
            radian=False,
        )
        self._sample_count += 1
        flags = FLAG_IMU_VALID
        if self._sample_count >= self._settle_samples:
            flags |= FLAG_ATTITUDE_VALID

        return (
            yaw_rate_x10,
            pitch_rate_x10,
            int(round(getattr(angle, YAW_AXIS) * YAW_SIGN * 100.0)),
            int(round(getattr(angle, PITCH_AXIS) * PITCH_SIGN * 100.0)),
            flags,
        )

    def is_calibrated(self):
        return self._calibrated
