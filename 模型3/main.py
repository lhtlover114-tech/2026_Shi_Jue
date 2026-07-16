# -*- coding: utf-8 -*-
"""模型3圆心追踪，以及 MaixCAM2 到 MSPM0 的 200 Hz 通信链路。"""

from maix import app, display, time

from find_circle import FindRectCircle
from maixcam_link import MaixCamLink


UART_BAUDRATE = 460800
REPORT_INTERVAL_MS = 1000


def main():
    disp = display.Display()
    finder = FindRectCircle(disp)
    finder.debug_draw_rect = True
    # 绘制第三个圆的全部轮廓点耗时较高，控制器也不需要这些绘制结果。
    # 保留基本画面预览和中心误差线即可。
    finder.debug_draw_circle = False

    link = MaixCamLink(
        tx_pin="A21",
        device="/dev/ttyS4",
        baudrate=UART_BAUDRATE,
        period_us=5000,
        target_timeout_ms=200,
    )
    # 先完成 UART 映射和打开，再启动独立发送线程。
    # 如果 UART 初始化失败，视觉循环不会进入看似正常的运行状态。
    link.start()

    print("[+] Model 3 circle tracking started")
    print(
        "[+] UART4: A21 TX, {} 8N1, 32-byte V2 frame, nominal 200 Hz".format(
            UART_BAUDRATE
        )
    )
    print("[+] IMU fields: zero placeholders; IMU_VALID=0")

    report_tick = time.ticks_ms()
    frame_count = 0
    valid_count = 0
    last_tx_count = 0

    while not app.need_exit():
        results = finder.run()
        err_center = results[2]
        updated = results[4]
        frame_count += 1

        if updated:
            x_error = int(round(err_center[0]))
            y_error = int(round(err_center[1]))
            link.publish_target(x_error, y_error)
            valid_count += 1

        now_ms = time.ticks_ms()
        elapsed_ms = (now_ms - report_tick) & 0xFFFFFFFF
        if elapsed_ms >= REPORT_INTERVAL_MS:
            tx_count, write_errors, skipped_slots = link.get_stats()
            vision_fps = frame_count * 1000.0 / elapsed_ms
            tx_hz = (tx_count - last_tx_count) * 1000.0 / elapsed_ms
            print(
                "[status] vision={:.1f}fps valid={} tx={:.1f}Hz "
                "write_err={} skipped={}".format(
                    vision_fps,
                    valid_count,
                    tx_hz,
                    write_errors,
                    skipped_slots,
                )
            )
            report_tick = now_ms
            frame_count = 0
            valid_count = 0
            last_tx_count = tx_count

    print("[+] exited")


if __name__ == "__main__":
    main()
