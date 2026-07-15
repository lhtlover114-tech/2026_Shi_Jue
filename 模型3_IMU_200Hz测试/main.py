# -*- coding: utf-8 -*-
"""独立验证模型3板载 IMU 能否维持 200 Hz 姿态采样与控制台输出。"""

from imu_attitude import ImuAttitude


PERIOD_US = 5000
REPORT_INTERVAL_US = 1000000
CALIBRATION_SAVE_ID = "model3_gimbal"


def format_sample(sequence, fields):
    yaw_rate_x10, pitch_rate_x10, yaw_x100, pitch_x100, flags = fields
    return (
        "imu[{:05d}] yr={:+.1f} pr={:+.1f} "
        "ya={:+.2f} pa={:+.2f} flags=0x{:02X}"
    ).format(
        int(sequence),
        yaw_rate_x10 / 10.0,
        pitch_rate_x10 / 10.0,
        yaw_x100 / 100.0,
        pitch_x100 / 100.0,
        int(flags) & 0xFF,
    )


def run_console_test(
    attitude,
    app_module,
    time_module,
    print_fn=print,
    period_us=PERIOD_US,
    report_interval_us=REPORT_INTERVAL_US,
):
    try:
        if not attitude.start():
            print_fn(
                "[imu-test] calibration {} not found; "
                "run the calibration app first".format(CALIBRATION_SAVE_ID)
            )
            return (0, 0, 0, 0)
    except Exception as exc:
        print_fn("[imu-test] initialization failed: {}".format(exc))
        return (0, 0, 1, 0)

    print_fn(
        "[imu-test] started: period={}us, console=every sample".format(
            period_us
        )
    )
    sample_count = 0
    print_count = 0
    error_count = 0
    skipped_count = 0
    sequence = 0
    next_deadline_us = time_module.ticks_us()
    report_start_us = next_deadline_us
    report_sample_count = 0
    report_print_count = 0

    while not app_module.need_exit():
        now_us = time_module.ticks_us()
        if now_us < next_deadline_us:
            time_module.sleep_us(next_deadline_us - now_us)
        now_us = time_module.ticks_us()

        elapsed_us = now_us - report_start_us
        if elapsed_us >= report_interval_us:
            elapsed_s = elapsed_us / 1000000.0
            print_fn(
                "[rate] sample={:.1f}Hz print={:.1f}Hz "
                "errors={} skipped={}".format(
                    (sample_count - report_sample_count) / elapsed_s,
                    (print_count - report_print_count) / elapsed_s,
                    error_count,
                    skipped_count,
                )
            )
            report_start_us = now_us
            report_sample_count = sample_count
            report_print_count = print_count

        try:
            fields = attitude.sample()
            sample_count += 1
            print_fn(format_sample(sequence, fields))
            print_count += 1
        except Exception as exc:
            error_count += 1
            if error_count == 1 or error_count % 200 == 0:
                print_fn("[imu-test] sample failed: {}".format(exc))

        sequence += 1
        next_deadline_us += period_us
        now_us = time_module.ticks_us()
        while next_deadline_us <= now_us:
            next_deadline_us += period_us
            skipped_count += 1

    return (sample_count, print_count, error_count, skipped_count)


def main():
    from maix import app, time

    run_console_test(ImuAttitude(), app, time)


if __name__ == "__main__":
    main()
