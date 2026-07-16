# -*- coding: utf-8 -*-
"""MaixCAM2 向 MSPM0 发送校准后六轴原始数据的固定帧链路。"""

import struct


FRAME_HEADER_0 = 0xA5
FRAME_HEADER_1 = 0x5A
FRAME_VERSION = 0x02
FRAME_SIZE = 32
CRC_DATA_SIZE = 30

FLAG_TARGET_VALID = 1 << 0
FLAG_IMU_VALID = 1 << 1

_FRAME_FORMAT = "<BBBBHHIhhhhhhhhH"


def _clamp_int16(value):
    value = int(value)
    if value > 32767:
        return 32767
    if value < -32768:
        return -32768
    return value


def _build_crc16_table():
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


_CRC16_TABLE = _build_crc16_table()


def crc16_ccitt_false(data):
    crc = 0xFFFF
    table = _CRC16_TABLE
    for byte in data:
        crc = ((crc << 8) ^ table[((crc >> 8) ^ byte) & 0xFF]) & 0xFFFF
    return crc


def build_frame(
    packet_sequence,
    vision_frame,
    timestamp_ms,
    x_error,
    y_error,
    acc_x_mg=0,
    acc_y_mg=0,
    acc_z_mg=0,
    gyro_x_dps_x10=0,
    gyro_y_dps_x10=0,
    gyro_z_dps_x10=0,
    flags=0,
):
    """生成32字节小端帧，末尾为CRC16-CCITT-FALSE。"""
    payload = struct.pack(
        _FRAME_FORMAT,
        FRAME_HEADER_0,
        FRAME_HEADER_1,
        FRAME_VERSION,
        FRAME_SIZE,
        int(packet_sequence) & 0xFFFF,
        int(vision_frame) & 0xFFFF,
        int(timestamp_ms) & 0xFFFFFFFF,
        _clamp_int16(x_error),
        _clamp_int16(y_error),
        _clamp_int16(acc_x_mg),
        _clamp_int16(acc_y_mg),
        _clamp_int16(acc_z_mg),
        _clamp_int16(gyro_x_dps_x10),
        _clamp_int16(gyro_y_dps_x10),
        _clamp_int16(gyro_z_dps_x10),
        int(flags) & 0xFFFF,
    )
    return payload + struct.pack("<H", crc16_ccitt_false(payload))


def resolve_target_snapshot(snapshot, now_ms, timeout_ms):
    if snapshot is None:
        return 0, 0, 0, 0

    vision_frame, x_error, y_error, updated_ms = snapshot
    age_ms = (int(now_ms) - int(updated_ms)) & 0xFFFFFFFF
    if age_ms >= int(timeout_ms):
        return vision_frame, 0, 0, 0
    return vision_frame, x_error, y_error, FLAG_TARGET_VALID


class MaixCamLink:
    """独占UART4，并按5ms周期读取和发送一次六轴数据。"""

    def __init__(
        self,
        tx_pin="A21",
        device="/dev/ttyS4",
        baudrate=460800,
        period_us=5000,
        target_timeout_ms=200,
        imu_source=None,
    ):
        self._tx_pin = tx_pin
        self._device = device
        self._baudrate = baudrate
        self._period_us = period_us
        self._target_timeout_ms = target_timeout_ms
        self._imu_source = imu_source

        self._vision_frame = 0
        self._target_snapshot = None
        self._stats = (0, 0, 0)
        self._imu_stats = (0, 0, False)
        self._timing_stats = (0,) * 11
        self._started = False
        self._serial = None
        self._time = None
        self._app = None
        self._thread = None

    def start(self):
        if self._started:
            return

        from maix import app, err, pinmap, thread, time, uart

        err.check_raise(
            pinmap.set_pin_function(self._tx_pin, "UART4_TX"),
            "Failed to map {} to UART4_TX".format(self._tx_pin),
        )
        self._serial = uart.UART(self._device, self._baudrate)
        self._time = time
        self._app = app
        self._started = True
        self._start_imu_source()

        worker = thread.Thread(self._tx_worker)
        self._thread = worker
        worker.detach()

    def publish_target(self, x_error, y_error):
        if not self._started:
            raise RuntimeError("MaixCamLink.start() must be called first")

        self._vision_frame = (self._vision_frame + 1) & 0xFFFF
        self._target_snapshot = (
            self._vision_frame,
            _clamp_int16(x_error),
            _clamp_int16(y_error),
            self._time.ticks_ms() & 0xFFFFFFFF,
        )

    def get_stats(self):
        return self._stats

    def get_imu_stats(self):
        return self._imu_stats

    def get_timing_stats(self):
        return self._timing_stats

    def _start_imu_source(self):
        if self._imu_source is None:
            return False

        samples, errors, _ = self._imu_stats
        try:
            calibrated = bool(self._imu_source.start())
        except Exception as exc:
            self._imu_stats = (samples, errors + 1, False)
            print("[imu] initialization failed:", exc)
            return False

        self._imu_stats = (samples, errors, calibrated)
        if not calibrated:
            print("[imu] calibration not found; raw fields are disabled")
        return calibrated

    def _read_imu_fields(self):
        if self._imu_source is None:
            return (self._time.ticks_ms(), 0, 0, 0, 0, 0, 0, 0)

        samples, errors, calibrated = self._imu_stats
        try:
            fields = self._imu_source.sample()
            samples += 1
            calibrated = bool(self._imu_source.is_calibrated())
            self._imu_stats = (samples, errors, calibrated)
            return fields
        except Exception as exc:
            errors += 1
            self._imu_stats = (samples, errors, calibrated)
            if errors == 1 or errors % 200 == 0:
                print("[imu] sample exception:", exc)
            return (self._time.ticks_ms(), 0, 0, 0, 0, 0, 0, 0)

    def _tx_worker(self, _):
        cycle_count = 0
        late_total_us = 0
        late_max_us = 0
        read_total_us = 0
        read_max_us = 0
        build_total_us = 0
        build_max_us = 0
        write_total_us = 0
        write_max_us = 0
        loop_total_us = 0
        loop_max_us = 0
        packet_sequence = 0
        sent_count = 0
        write_error_count = 0
        skipped_slot_count = 0
        next_deadline_us = self._time.ticks_us()

        while not self._app.need_exit():
            now_us = self._time.ticks_us()
            if now_us < next_deadline_us:
                self._time.sleep_us(next_deadline_us - now_us)

            cycle_start_us = self._time.ticks_us()
            late_us = max(0, cycle_start_us - next_deadline_us)
            read_start_us = cycle_start_us
            (
                timestamp_ms,
                acc_x_mg,
                acc_y_mg,
                acc_z_mg,
                gyro_x_dps_x10,
                gyro_y_dps_x10,
                gyro_z_dps_x10,
                imu_flags,
            ) = self._read_imu_fields()
            read_end_us = self._time.ticks_us()

            vision_frame, x_error, y_error, target_flags = (
                resolve_target_snapshot(
                    self._target_snapshot,
                    timestamp_ms,
                    self._target_timeout_ms,
                )
            )
            frame = build_frame(
                packet_sequence=packet_sequence,
                vision_frame=vision_frame,
                timestamp_ms=timestamp_ms,
                x_error=x_error,
                y_error=y_error,
                acc_x_mg=acc_x_mg,
                acc_y_mg=acc_y_mg,
                acc_z_mg=acc_z_mg,
                gyro_x_dps_x10=gyro_x_dps_x10,
                gyro_y_dps_x10=gyro_y_dps_x10,
                gyro_z_dps_x10=gyro_z_dps_x10,
                flags=target_flags | imu_flags,
            )
            build_end_us = self._time.ticks_us()

            try:
                written = self._serial.write(frame)
            except Exception as exc:
                written = -1
                if write_error_count == 0 or write_error_count % 200 == 0:
                    print("[link] UART4 write exception:", exc)
            write_end_us = self._time.ticks_us()

            sent_count += 1
            packet_sequence = (packet_sequence + 1) & 0xFFFF
            if written != FRAME_SIZE:
                write_error_count += 1
                if write_error_count == 1 or write_error_count % 200 == 0:
                    print("[link] UART4 short write:", written)

            read_us = read_end_us - read_start_us
            build_us = build_end_us - read_end_us
            write_us = write_end_us - build_end_us
            loop_us = write_end_us - cycle_start_us
            cycle_count += 1
            late_total_us += late_us
            late_max_us = max(late_max_us, late_us)
            read_total_us += read_us
            read_max_us = max(read_max_us, read_us)
            build_total_us += build_us
            build_max_us = max(build_max_us, build_us)
            write_total_us += write_us
            write_max_us = max(write_max_us, write_us)
            loop_total_us += loop_us
            loop_max_us = max(loop_max_us, loop_us)
            self._timing_stats = (
                cycle_count,
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
            )

            next_deadline_us += self._period_us
            now_us = self._time.ticks_us()
            while next_deadline_us <= now_us:
                next_deadline_us += self._period_us
                skipped_slot_count += 1

            self._stats = (sent_count, write_error_count, skipped_slot_count)
