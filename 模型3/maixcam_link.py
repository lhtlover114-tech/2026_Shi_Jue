# -*- coding: utf-8 -*-
"""MaixCAM2 -> MSPM0 fixed-frame vision link."""

import struct


FRAME_HEADER_0 = 0xA5
FRAME_HEADER_1 = 0x5A
FRAME_VERSION = 0x01
FRAME_SIZE = 28
CRC_DATA_SIZE = 26

FLAG_TARGET_VALID = 1 << 0
FLAG_IMU_VALID = 1 << 1
FLAG_ATTITUDE_VALID = 1 << 2

_FRAME_FORMAT = "<BBBBHHIhhhhhhH"


def _clamp_int16(value):
    value = int(value)
    if value > 32767:
        return 32767
    if value < -32768:
        return -32768
    return value


def crc16_ccitt_false(data):
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, xorout 0."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def build_frame(
    packet_sequence,
    vision_frame,
    timestamp_ms,
    x_error,
    y_error,
    yaw_rate_x10=0,
    pitch_rate_x10=0,
    yaw_angle_x100=0,
    pitch_angle_x100=0,
    flags=0,
):
    """Build one 28-byte little-endian frame matching maixcam_link.c."""
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
        _clamp_int16(yaw_rate_x10),
        _clamp_int16(pitch_rate_x10),
        _clamp_int16(yaw_angle_x100),
        _clamp_int16(pitch_angle_x100),
        int(flags) & 0xFFFF,
    )
    return payload + struct.pack("<H", crc16_ccitt_false(payload))


def resolve_target_snapshot(snapshot, now_ms, timeout_ms):
    """Return vision frame, X/Y and flags for the current send time."""
    if snapshot is None:
        return 0, 0, 0, 0

    vision_frame, x_error, y_error, updated_ms = snapshot
    age_ms = (int(now_ms) - int(updated_ms)) & 0xFFFFFFFF
    if age_ms >= int(timeout_ms):
        return vision_frame, 0, 0, 0
    return vision_frame, x_error, y_error, FLAG_TARGET_VALID


class MaixCamLink:
    """Own UART4 and transmit the latest vision snapshot at nominal 200 Hz."""

    def __init__(
        self,
        tx_pin="A21",
        device="/dev/ttyS4",
        baudrate=115200,
        period_us=5000,
        target_timeout_ms=200,
        attitude_source=None,
    ):
        self._tx_pin = tx_pin
        self._device = device
        self._baudrate = baudrate
        self._period_us = period_us
        self._target_timeout_ms = target_timeout_ms
        self._attitude_source = attitude_source

        self._vision_frame = 0
        self._target_snapshot = None
        self._stats = (0, 0, 0)  # sent attempts, write errors, skipped slots
        self._imu_stats = (0, 0, False)  # 采样次数、错误数、校准是否有效
        self._started = False
        self._serial = None
        self._time = None
        self._app = None
        self._thread = None

    def start(self):
        """Map A21, open UART4, then start the detached transmit worker."""
        if self._started:
            return

        from maix import app, err, pinmap, thread, time, uart

        err.check_raise(
            pinmap.set_pin_function(self._tx_pin, "UART4_TX"),
            "Failed to map {} to UART4_TX".format(self._tx_pin),
        )
        # UART defaults are 8 data bits, no parity, 1 stop bit, no flow control.
        serial = uart.UART(self._device, self._baudrate)

        self._serial = serial
        self._time = time
        self._app = app
        self._started = True
        self._start_attitude_source()

        worker = thread.Thread(self._tx_worker)
        self._thread = worker
        worker.detach()

    def publish_target(self, x_error, y_error):
        """Publish one new valid visual measurement to the transmit worker."""
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
        """Return sent attempts, UART write errors and skipped 5 ms slots."""
        return self._stats

    def get_imu_stats(self):
        """返回 IMU 采样次数、错误数和校准是否有效。"""
        return self._imu_stats

    def _start_attitude_source(self):
        if self._attitude_source is None:
            return False

        samples, errors, _ = self._imu_stats
        try:
            calibrated = bool(self._attitude_source.start())
        except Exception as exc:
            self._imu_stats = (samples, errors + 1, False)
            print("[imu] initialization failed:", exc)
            return False

        self._imu_stats = (samples, errors, calibrated)
        if not calibrated:
            print("[imu] calibration not found; IMU fields are disabled")
        return calibrated

    def _read_attitude_fields(self):
        if self._attitude_source is None:
            return (0, 0, 0, 0, 0)

        samples, errors, calibrated = self._imu_stats
        try:
            fields = self._attitude_source.sample()
            samples += 1
            calibrated = bool(self._attitude_source.is_calibrated())
            self._imu_stats = (samples, errors, calibrated)
            return fields
        except Exception as exc:
            errors += 1
            self._imu_stats = (samples, errors, calibrated)
            if errors == 1 or errors % 200 == 0:
                print("[imu] sample exception:", exc)
            return (0, 0, 0, 0, 0)

    def _tx_worker(self, _):
        packet_sequence = 0
        sent_count = 0
        write_error_count = 0
        skipped_slot_count = 0
        next_deadline_us = self._time.ticks_us()

        while not self._app.need_exit():
            now_us = self._time.ticks_us()
            if now_us < next_deadline_us:
                self._time.sleep_us(next_deadline_us - now_us)

            timestamp_ms = self._time.ticks_ms() & 0xFFFFFFFF
            vision_frame, x_error, y_error, flags = resolve_target_snapshot(
                self._target_snapshot,
                timestamp_ms,
                self._target_timeout_ms,
            )
            (
                yaw_rate_x10,
                pitch_rate_x10,
                yaw_angle_x100,
                pitch_angle_x100,
                imu_flags,
            ) = self._read_attitude_fields()
            frame = build_frame(
                packet_sequence=packet_sequence,
                vision_frame=vision_frame,
                timestamp_ms=timestamp_ms,
                x_error=x_error,
                y_error=y_error,
                yaw_rate_x10=yaw_rate_x10,
                pitch_rate_x10=pitch_rate_x10,
                yaw_angle_x100=yaw_angle_x100,
                pitch_angle_x100=pitch_angle_x100,
                flags=flags | imu_flags,
            )

            try:
                written = self._serial.write(frame)
            except Exception as exc:
                written = -1
                if write_error_count == 0 or write_error_count % 200 == 0:
                    print("[link] UART4 write exception:", exc)

            sent_count += 1
            packet_sequence = (packet_sequence + 1) & 0xFFFF
            if written != FRAME_SIZE:
                write_error_count += 1
                if write_error_count == 1 or write_error_count % 200 == 0:
                    print("[link] UART4 short write:", written)

            next_deadline_us += self._period_us
            now_us = self._time.ticks_us()
            while next_deadline_us <= now_us:
                next_deadline_us += self._period_us
                skipped_slot_count += 1

            self._stats = (sent_count, write_error_count, skipped_slot_count)

