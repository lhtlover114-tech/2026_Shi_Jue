# -*- coding: utf-8 -*-
"""独立验证校准后六轴读取与 UART4 发送能否维持 200 Hz。"""

from maixcam_link import FLAG_IMU_VALID, MaixCamLink


UART_BAUDRATE = 460800
PERIOD_US = 5000
REPORT_INTERVAL_MS = 1000
CALIBRATION_SAVE_ID = "model3_gimbal"


class RawImuSource:
    """读取已应用陀螺仪零偏、尚未姿态融合的六轴数据。"""

    def __init__(self, sensor=None, time_module=None):
        self._sensor = sensor
        self._time = time_module
        self._calibrated = False

    def _create_hardware(self):
        from maix import time
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

    def start(self):
        if self._calibrated:
            return True
        if self._sensor is None:
            self._create_hardware()

        if not self._sensor.calib_gyro_exists(CALIBRATION_SAVE_ID):
            return False

        self._sensor.load_calib_gyro(CALIBRATION_SAVE_ID)
        self._calibrated = True
        return True

    def sample(self):
        """返回时间戳、三轴加速度 mg、三轴角速度 0.1 dps 和有效标志。"""
        if not self._calibrated:
            return (self._time.ticks_ms(), 0, 0, 0, 0, 0, 0, 0)

        data = self._sensor.read_all(calib_gryo=True, radian=False)
        timestamp_ms = self._time.ticks_ms() & 0xFFFFFFFF
        return (
            timestamp_ms,
            int(round(data.acc.x * 1000.0)),
            int(round(data.acc.y * 1000.0)),
            int(round(data.acc.z * 1000.0)),
            int(round(data.gyro.x * 10.0)),
            int(round(data.gyro.y * 10.0)),
            int(round(data.gyro.z * 10.0)),
            FLAG_IMU_VALID,
        )

    def is_calibrated(self):
        return self._calibrated


def main():
    from maix import app, time

    imu_source = RawImuSource()
    if not imu_source.start():
        print(
            "[imu-raw] calibration {} not found; run calibration first".format(
                CALIBRATION_SAVE_ID
            )
        )
        return

    link = MaixCamLink(
        tx_pin="A21",
        device="/dev/ttyS4",
        baudrate=UART_BAUDRATE,
        period_us=PERIOD_US,
        target_timeout_ms=200,
        imu_source=imu_source,
    )
    link.start()

    print("[+] Raw IMU 200 Hz test started")
    print(
        "[+] UART4: A21 TX, {} 8N1, frame=32 bytes".format(
            UART_BAUDRATE
        )
    )
    print("[+] IMU: read_all(calib_gryo=True, radian=False)")

    report_tick = time.ticks_ms()
    last_tx_count = 0
    last_imu_count = 0

    while not app.need_exit():
        time.sleep_ms(50)
        now_ms = time.ticks_ms()
        elapsed_ms = (now_ms - report_tick) & 0xFFFFFFFF
        if elapsed_ms < REPORT_INTERVAL_MS:
            continue

        tx_count, write_errors, skipped_slots = link.get_stats()
        imu_count, imu_errors, imu_calibrated = link.get_imu_stats()
        (
            timing_count,
            late_total_us,
            late_max_us,
            read_total_us,
            read_max_us,
            build_total_us,
            build_max_us,
            write_total_us,
            write_max_us,
            loop_total_us,
            loop_max_us,
        ) = link.get_timing_stats()
        denominator = max(1, timing_count)
        tx_hz = (tx_count - last_tx_count) * 1000.0 / elapsed_ms
        imu_hz = (imu_count - last_imu_count) * 1000.0 / elapsed_ms

        print(
            "[status] raw={:.1f}Hz tx={:.1f}Hz imu_err={} imu_cal={} "
            "write_err={} skipped={}".format(
                imu_hz,
                tx_hz,
                imu_errors,
                int(imu_calibrated),
                write_errors,
                skipped_slots,
            )
        )
        print(
            "[timing] late={:.0f}/{}us read={:.0f}/{}us "
            "build={:.0f}/{}us write={:.0f}/{}us loop={:.0f}/{}us".format(
                late_total_us / denominator,
                late_max_us,
                read_total_us / denominator,
                read_max_us,
                build_total_us / denominator,
                build_max_us,
                write_total_us / denominator,
                write_max_us,
                loop_total_us / denominator,
                loop_max_us,
            )
        )

        report_tick = now_ms
        last_tx_count = tx_count
        last_imu_count = imu_count


if __name__ == "__main__":
    main()
