# -*- coding: utf-8 -*-
"""
巡线 UART 通信模块：MaixCAM2 ↔ MSPM0
复用在用的 32 字节 V2 帧协议，460800 baud / 100Hz。

架构：
  - TX 线程（独立）：每 period_us 发送最新巡线数据，不阻塞视觉主循环
  - 主线程（视觉） ：调用 publish_line_data() 存入快照，瞬时返回
  - RX 线程（可选）：监听 MSPM0 发来的指令（速度/模式切换等）

巡线字段映射：
  - x_error: near_error（近处条带误差）
  - y_error: far_error（远处条带误差）

引脚：
  UART4_TX → A21    连接 MSPM0 RX
  UART4_RX → A22    连接 MSPM0 TX（可选，双向通信时启用）
"""

import struct

from maix import app, err, pinmap, thread, time, uart

# ======================== 协议常量 ========================

FRAME_HEADER_0 = 0xA5
FRAME_HEADER_1 = 0x5A
FRAME_VERSION = 0x02
FRAME_SIZE = 32
CRC_DATA_SIZE = 30

FLAG_TARGET_VALID = 1 << 0   # bit0: 巡线数据有效
FLAG_IMU_VALID = 1 << 1      # bit1: IMU 数据有效（巡线暂不使用）

_FRAME_FORMAT = "<BBBBHHIhhhhhhhhH"


# ======================== 工具函数 ========================

def _clamp_int16(value):
    """将值夹到 int16 范围 [-32768, 32767]"""
    value = int(value)
    if value > 32767:
        return 32767
    if value < -32768:
        return -32768
    return value


def crc16_ccitt_false(data):
    """CRC-16/CCITT-FALSE（多项式 0x1021，初值 0xFFFF）"""
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
    y_error=0,
    acc_x_mg=0, acc_y_mg=0, acc_z_mg=0,
    gyro_x_dps_x10=0, gyro_y_dps_x10=0, gyro_z_dps_x10=0,
    flags=0,
):
    """
    生成 32 字节 V2 小端帧。

    巡线模式下 x_error=near_error，y_error=far_error，IMU 字段填 0。
    """
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


# ======================== 通信类 ========================

class LineFollowLink:
    """
    巡线专用 UART 通信链路。

    使用方法:
        link = LineFollowLink()
        link.DEBUG_PRINT = True
        link.start()

        while True:
            result = follower.process()
            link.publish_line_data(
                near_error=result['near_error'],
                far_error=result['far_error'],
                confidence=result['confidence'],
            )
    """

    # 调试打印（可运行时切换）
    DEBUG_PRINT = False           # 是否在控制台打印发送数据
    DEBUG_PRINT_INTERVAL = 40     # 每 N 帧打印一次（200Hz 下 40=200ms）

    def __init__(
        self,
        tx_pin="A21",
        rx_pin="A22",
        device="/dev/ttyS4",
        baudrate=460800,
        period_us=10000,          # 100Hz 发送周期，改为5000即200Hz
        enable_rx=False,          # 是否启用接收（双向通信）
    ):
        self._tx_pin = tx_pin
        self._rx_pin = rx_pin
        self._device = device
        self._baudrate = baudrate
        self._period_us = period_us
        self._enable_rx = enable_rx

        # 状态
        self._vision_frame = 0
        # (vision_frame, near_error, far_error, flags, fps)
        self._target_snapshot = None
        self._stats = (0, 0, 0)         # (sent, errors, skipped)
        self._started = False
        self._serial = None
        self._rx_buffer = b""           # 接收缓冲
        self._rx_callbacks = []         # 接收回调列表

        self._time = None
        self._app = None
        self._tx_thread = None
        self._rx_thread = None

    # ==================== 启动 / 停止 ====================

    def start(self):
        """初始化 UART4、映射引脚、启动收发线程"""
        if self._started:
            return

        # 映射 TX 引脚
        err.check_raise(
            pinmap.set_pin_function(self._tx_pin, "UART4_TX"),
            f"Failed to map {self._tx_pin} to UART4_TX",
        )

        # 如果启用 RX，映射 RX 引脚
        if self._enable_rx:
            err.check_raise(
                pinmap.set_pin_function(self._rx_pin, "UART4_RX"),
                f"Failed to map {self._rx_pin} to UART4_RX",
            )

        self._serial = uart.UART(self._device, self._baudrate)
        self._time = time
        self._app = app
        self._started = True

        # 启动 TX 发送线程
        tx_worker = thread.Thread(self._tx_worker)
        self._tx_thread = tx_worker
        tx_worker.detach()

        # 如果启用 RX，启动接收线程
        if self._enable_rx:
            rx_worker = thread.Thread(self._rx_worker)
            self._rx_thread = rx_worker
            rx_worker.detach()

        print(f"[link] started (TX={self._tx_pin}" +
              (f", RX={self._rx_pin}" if self._enable_rx else "") +
              f", baud={self._baudrate}, period={self._period_us}us)")

    # ==================== 数据发布（主线程调用） ====================

    def publish_line_data(
        self,
        near_error,
        confidence=1.0,
        fps=0.0,
        far_error=0.0,
    ):
        """
        发布最新巡线数据（主线程/视觉循环调用，瞬时返回）。

        参数:
            near_error: 近处线偏移量 (px)，正=右偏, 负=左偏
            confidence: 整体置信度 0.0~1.0，< 0.3 视为无效
            fps:        视觉帧率（调试用，不发送给 MSPM0）
            far_error:  远处线偏移量 (px)，正=右偏, 负=左偏

        为兼容旧调用，第二和第三个位置参数仍分别是 confidence、fps；
        far_error 建议使用关键字参数传入。
        """
        if not self._started:
            raise RuntimeError("LineFollowLink.start() must be called first")

        self._vision_frame = (self._vision_frame + 1) & 0xFFFF

        flags = FLAG_TARGET_VALID if confidence >= 0.3 else 0

        self._target_snapshot = (
            self._vision_frame,
            _clamp_int16(near_error),
            _clamp_int16(far_error),
            flags,
            fps,    # 视觉帧率（仅调试打印用）
        )

    def publish_line_lost(self, fps=0.0):
        """显式告知 MSPM0 线已丢失（FLAG_TARGET_VALID = 0）"""
        self._vision_frame = (self._vision_frame + 1) & 0xFFFF
        self._target_snapshot = (self._vision_frame, 0, 0, 0, fps)

    # ==================== RX 回调 ====================

    def on_rx(self, callback):
        """
        注册 MSPM0 → MaixCAM2 数据接收回调。
        callback(bytes) 接收来自 MSPM0 的原始数据帧。
        """
        self._rx_callbacks.append(callback)

    # ==================== 状态查询 ====================

    def get_stats(self):
        """返回 (发送次数, 写入错误数, 跳过周期数)"""
        return self._stats

    # ==================== TX 发送线程 ====================

    def _tx_worker(self, _):
        """独立线程：每 period_us 发送一帧，与视觉循环完全解耦"""
        packet_sequence = 0
        sent_count = 0
        write_error_count = 0
        skipped_slot_count = 0
        next_deadline_us = self._time.ticks_us()
        t_start_ms = self._time.ticks_ms()  # 用于计算实际线程发送频率

        while not self._app.need_exit():
            # 等待到下一个发送时刻
            now_us = self._time.ticks_us()
            if now_us < next_deadline_us:
                self._time.sleep_us(next_deadline_us - now_us)

            timestamp_ms = self._time.ticks_ms() & 0xFFFFFFFF

            # 读取最新快照
            snapshot = self._target_snapshot
            if snapshot is None:
                vision_frame = 0
                near_error = 0
                far_error = 0
                flags = 0
                visual_fps = 0.0
            else:
                vision_frame, near_error, far_error, flags, visual_fps = snapshot

            # 构建帧：x_error=near_error, y_error=far_error
            frame = build_frame(
                packet_sequence=packet_sequence,
                vision_frame=vision_frame,
                timestamp_ms=timestamp_ms,
                x_error=near_error,
                y_error=far_error,
                flags=flags,
            )

            # 发送
            try:
                written = self._serial.write(frame)
            except Exception as exc:
                written = -1
                if write_error_count == 0 or write_error_count % 200 == 0:
                    print(f"[link] write err: {exc}")

            sent_count += 1
            packet_sequence = (packet_sequence + 1) & 0xFFFF

            # 调试打印（每 N 帧输出一次，避免刷屏）
            if self.DEBUG_PRINT and sent_count % self.DEBUG_PRINT_INTERVAL == 0:
                elapsed_s = (self._time.ticks_ms() - t_start_ms) / 1000.0
                comm_freq = sent_count / elapsed_s if elapsed_s > 0 else 0
                valid = "V" if flags & FLAG_TARGET_VALID else "X"
                print(f"[link] seq={packet_sequence:5d}  "
                      f"near={near_error:+5d}  "
                      f"far={far_error:+5d}  "
                      f"vis_fps={visual_fps:4.1f}  "
                      f"tx_freq={comm_freq:5.1f}Hz  "
                      f"flags={flags:#06x}({valid})  "
                      f"frm={vision_frame}")

            if written != FRAME_SIZE:
                write_error_count += 1
                if write_error_count == 1 or write_error_count % 200 == 0:
                    print(f"[link] short write: {written}")

            # 维护发送节拍，处理周期溢出
            next_deadline_us += self._period_us
            now_us = self._time.ticks_us()
            while next_deadline_us <= now_us:
                next_deadline_us += self._period_us
                skipped_slot_count += 1

            self._stats = (sent_count, write_error_count, skipped_slot_count)

    # ==================== RX 接收线程（可选） ====================

    def _rx_worker(self, _):
        """独立线程：持续监听 MSPM0 发来的数据"""
        while not self._app.need_exit():
            try:
                available = self._serial.any()
            except Exception:
                self._time.sleep_ms(10)
                continue

            if available > 0:
                try:
                    data = self._serial.read(available)
                except Exception:
                    self._time.sleep_ms(10)
                    continue

                if data:
                    self._rx_buffer += data

                    # 按帧头同步
                    while len(self._rx_buffer) >= 2:
                        idx = self._rx_buffer.find(
                            bytes([FRAME_HEADER_0, FRAME_HEADER_1])
                        )
                        if idx < 0:
                            self._rx_buffer = b""  # 全丢弃
                            break
                        if idx > 0:
                            self._rx_buffer = self._rx_buffer[idx:]
                            continue

                        # 等待完整帧
                        if len(self._rx_buffer) < FRAME_SIZE:
                            break

                        frame = self._rx_buffer[:FRAME_SIZE]
                        self._rx_buffer = self._rx_buffer[FRAME_SIZE:]

                        # 通知回调
                        for cb in self._rx_callbacks:
                            try:
                                cb(frame)
                            except Exception:
                                pass
            else:
                self._time.sleep_ms(5)
