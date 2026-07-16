# -*- coding: utf-8 -*-
"""MaixCAM2到MSPM0的视觉与六轴占位字段固定帧链路。"""

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


def crc16_ccitt_false(data):
    """计算CRC-16/CCITT-FALSE。"""
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
    acc_x_mg=0,
    acc_y_mg=0,
    acc_z_mg=0,
    gyro_x_dps_x10=0,
    gyro_y_dps_x10=0,
    gyro_z_dps_x10=0,
    flags=0,
):
    """生成32字节V2小端帧；本阶段六轴参数保持默认值0。"""
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


def resolve_target_snapshot(snapshot):
    """返回最新视觉帧号、X/Y误差和目标有效标志。"""
    if snapshot is None:
        return 0, 0, 0, 0

    vision_frame, x_error, y_error = snapshot
    return vision_frame, x_error, y_error, FLAG_TARGET_VALID


class MaixCamLink:
    """独占UART4，并以名义200Hz发送最新视觉快照。"""

    def __init__(
        self,
        tx_pin="A21",
        device="/dev/ttyS4",
        baudrate=460800,
        period_us=5000,
    ):
        self._tx_pin = tx_pin
        self._device = device
        self._baudrate = baudrate
        self._period_us = period_us

        self._vision_frame = 0
        self._target_snapshot = None
        self._stats = (0, 0, 0)
        self._started = False
        self._serial = None
        self._time = None
        self._app = None
        self._thread = None

    def start(self):
        """映射A21并启动独立UART4发送线程。"""
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

        worker = thread.Thread(self._tx_worker)
        self._thread = worker
        worker.detach()

    def publish_target(self, x_error, y_error):
        """向发送线程发布一组新的有效视觉测量值。"""
        if not self._started:
            raise RuntimeError("MaixCamLink.start() must be called first")

        self._vision_frame = (self._vision_frame + 1) & 0xFFFF
        self._target_snapshot = (
            self._vision_frame,
            _clamp_int16(x_error),
            _clamp_int16(y_error),
        )

    def get_stats(self):
        """返回发送次数、UART写入错误数和跳过的5ms周期数。"""
        return self._stats

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
                self._target_snapshot
            )
            frame = build_frame(
                packet_sequence=packet_sequence,
                vision_frame=vision_frame,
                timestamp_ms=timestamp_ms,
                x_error=x_error,
                y_error=y_error,
                flags=flags,
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
