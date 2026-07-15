# -*- coding: utf-8 -*-
"""模型3圆心追踪，以及 MaixCAM2 到 MSPM0 的 200 Hz 通信链路。"""

from maix import app, display, time

from find_circle import FindRectCircle
from imu_attitude import ImuAttitude
from maixcam_link import MaixCamLink


REPORT_INTERVAL_MS = 1000


def main():
    disp = display.Display()
    finder = FindRectCircle(disp)
    #finder.debug_draw_rect = True
    # 绘制第三个圆的全部轮廓点耗时较高，控制器也不需要这些绘制结果。
    # 保留基本画面预览和中心误差线即可。
    finder.debug_draw_circle = False

    attitude = ImuAttitude()
    link = MaixCamLink(
        tx_pin="A21",
        device="/dev/ttyS4",
        baudrate=115200,
        period_us=5000,
        target_timeout_ms=200,
        attitude_source=attitude,
    )
    # 先完成 UART 映射和打开，再启动独立发送线程。
    # 如果 UART 初始化失败，视觉循环不会进入看似正常的运行状态。
    link.start()

    print("[+] Model 3 circle tracking started")
    print("[+] UART4: A21 TX, 115200 8N1, nominal 200 Hz")

    report_tick = time.ticks_ms()
    frame_count = 0
    valid_count = 0
    last_tx_count = 0
    last_imu_sample_count = 0

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
            imu_sample_count, imu_errors, imu_calibrated = link.get_imu_stats()
            vision_fps = frame_count * 1000.0 / elapsed_ms
            tx_hz = (tx_count - last_tx_count) * 1000.0 / elapsed_ms
            imu_hz = (
                (imu_sample_count - last_imu_sample_count)
                * 1000.0
                / elapsed_ms
            )
            print(
                "[status] vision={:.1f}fps valid={} tx={:.1f}Hz "
                "imu={:.1f}Hz imu_err={} imu_cal={} "
                "write_err={} skipped={}".format(
                    vision_fps,
                    valid_count,
                    tx_hz,
                    imu_hz,
                    imu_errors,
                    int(imu_calibrated),
                    write_errors,
                    skipped_slots,
                )
            )
            report_tick = now_ms
            frame_count = 0
            valid_count = 0
            last_tx_count = tx_count
            last_imu_sample_count = imu_sample_count

    print("[+] exited")


if __name__ == "__main__":
    main()
