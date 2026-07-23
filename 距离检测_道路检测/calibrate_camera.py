"""相机标定脚本：用已知距离的实际靶标定 fx/fy。

原理 —— 小孔成像模型：fx = 像素宽度 × 距离 / 靶实际宽度

使用方法：
  1. 修改下方 CALIBRATION_DISTANCES_MM 为你要测量的实际距离（mm）
  2. 在 MaixCAM2 上运行此脚本
  3. 按照屏幕提示，把 29.7cm×21cm 靶依次放在每个距离处
  4. 每个距离保持稳定，脚本自动采集若干帧后跳到下一个距离
  5. 标定结果自动保存到 camera_calib.json
  6. 重新运行 find_circle_pose.py 即可使用标定后的内参
"""

import math
import os

import cv2
import numpy as np

from maix import camera, display, image, time

# 复用 find_circle_pose 中的检测函数和常量
from find_circle_pose import (
    detect_rectangle,
    scale_points,
    target_pixel_width,
    distance_2d,
    order_quad_points,
    save_camera_params,
    CALIB_FILE,
    CAM_W,
    CAM_H,
    PROC_W,
    PROC_H,
    TARGET_W_MM,
    TARGET_H_MM,
    CAMERA_CX,
    CAMERA_CY,
    DIST_COEFFS,
)

# ============================================================
# 标定配置 —— 根据实际测量环境修改此列表
# ============================================================

# 已知距离列表 (mm)，把靶依次放在这些距离处。
# 建议覆盖比赛可能用到的全部距离范围，例如 0.5m ~ 3m。
CALIBRATION_DISTANCES_MM = [500, 1000, 1500, 2000, 3000]

# 每个距离的采集时长 (秒)。光线好时可以设短一些。
SAMPLING_DURATION_S = 3.0

# 每个距离最少需要的有效检测帧数。太少则该距离被跳过。
MIN_SAMPLES_PER_DISTANCE = 10

# 摄像头预热帧数。
WARMUP_FRAMES = 20


def compute_pixel_height(ordered_points):
    """计算靶在图像中的平均像素高度：(左边高 + 右边高) / 2。"""
    if ordered_points is None or len(ordered_points) != 4:
        return 0.0
    left_height = distance_2d(ordered_points[0], ordered_points[3])
    right_height = distance_2d(ordered_points[1], ordered_points[2])
    return (left_height + right_height) * 0.5


def collect_samples(cam, target_distance_mm, duration_s, disp):
    """在指定距离处采集若干 (像素宽度, 像素高度) 样本。

    Args:
        cam: 已初始化的 MaixCAM2 摄像头实例。
        target_distance_mm: 当前靶距 (仅用于屏幕显示)。
        duration_s: 采集时长 (秒)。
        disp: MaixCAM2 显示实例。

    Returns:
        list of (pixel_width, pixel_height) 元组。
    """
    samples = []
    start_ms = time.ticks_ms()
    duration_ms = int(duration_s * 1000)

    while True:
        elapsed = (time.ticks_ms() - start_ms) & 0xFFFFFFFF
        if elapsed >= duration_ms:
            break

        img = cam.read()
        if img is None:
            continue

        frame_bgr = image.image2cv(img, ensure_bgr=False, copy=False)
        process_frame = cv2.resize(
            frame_bgr, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA
        )
        process_points = detect_rectangle(process_frame)
        points = scale_points(process_points, PROC_W, PROC_H, CAM_W, CAM_H)

        if points is not None:
            ordered = order_quad_points(points)
            if ordered is not None:
                pw = target_pixel_width(ordered)
                ph = compute_pixel_height(ordered)
                if pw > 1.0 and ph > 1.0:
                    samples.append((pw, ph))

        # 在屏幕上叠加状态信息
        remaining_s = max(0.0, (duration_ms - elapsed) / 1000.0)
        img.draw_string(
            4, 4,
            "CALIB: {:4.0f}mm".format(target_distance_mm),
            image.COLOR_GREEN, scale=1.4, thickness=2,
        )
        img.draw_string(
            4, 28,
            "samples:{}  remain:{:.1f}s".format(len(samples), remaining_s),
            image.COLOR_YELLOW, scale=1.2, thickness=2,
        )
        if samples:
            pw, ph = samples[-1]
            img.draw_string(
                4, 50,
                "W:{:.1f}  H:{:.1f} px".format(pw, ph),
                image.COLOR_GREEN, scale=1.2, thickness=2,
            )
        disp.show(img)

    return samples


def compute_calibration(samples_by_distance):
    """根据各距离采集的样本计算标定后的 fx / fy。

    每个距离内取中位数抗干扰，各距离的 fx 取平均作为最终值。
    """
    print("\n" + "=" * 64)
    print("  {:>6s}  {:>7s}  {:>8s}  {:>8s}  {:>7s}  {:>7s}".format(
        "Dist", "Samples", "Med_W", "Med_H", "fx", "fy"))
    print("-" * 64)

    fx_values = []
    fy_values = []
    results = []

    for distance_mm in sorted(samples_by_distance.keys()):
        sample_list = samples_by_distance[distance_mm]
        if len(sample_list) < MIN_SAMPLES_PER_DISTANCE:
            print(
                "  {:6.0f}mm  {:3d}     {:>8s}".format(
                    distance_mm, len(sample_list), "SKIPPED")
            )
            continue

        widths = [s[0] for s in sample_list]
        heights = [s[1] for s in sample_list]
        median_w = sorted(widths)[len(widths) // 2]
        median_h = sorted(heights)[len(heights) // 2]

        fx_est = median_w * distance_mm / TARGET_W_MM
        fy_est = median_h * distance_mm / TARGET_H_MM

        fx_values.append(fx_est)
        fy_values.append(fy_est)
        results.append({
            "distance_mm": distance_mm,
            "median_pixel_width": round(median_w, 2),
            "median_pixel_height": round(median_h, 2),
            "fx_estimate": round(fx_est, 2),
            "fy_estimate": round(fy_est, 2),
            "sample_count": len(sample_list),
        })

        print(
            "  {:6.0f}mm  {:3d}     {:8.1f}  {:8.1f}  {:7.1f}  {:7.1f}".format(
                distance_mm, len(sample_list),
                median_w, median_h, fx_est, fy_est,
            )
        )

    print("-" * 64)

    if not fx_values:
        print("[calib] ERROR: no valid distance measurements!")
        return None

    fx_final = sum(fx_values) / len(fx_values)
    fy_final = sum(fy_values) / len(fy_values)

    print("  FINAL:  fx={:.2f}  fy={:.2f}".format(fx_final, fy_final))

    # 一致性检查：各距离反推出的 fx 应该接近
    if len(fx_values) >= 2:
        fx_range = max(fx_values) - min(fx_values)
        fx_variation = fx_range / fx_final * 100.0
        print("  fx variation across distances: {:.1f}%".format(fx_variation))
        if fx_variation > 5.0:
            print("  ** WARNING: fx varies >5%, re-check measurements! **")

    return fx_final, fy_final, results


def main():
    print("=" * 64)
    print("  Camera Calibration  --  Actual Distance Method")
    print("  Target: {:.0f} x {:.0f} mm".format(TARGET_W_MM, TARGET_H_MM))
    print("  Distances: {} mm".format(CALIBRATION_DISTANCES_MM))
    print("  Sampling: {:.1f}s per distance".format(SAMPLING_DURATION_S))
    print("=" * 64)

    disp = display.Display()

    # 初始化摄像头（与 find_circle_pose.py 一致的参数）
    cam = camera.Camera(
        CAM_W, CAM_H,
        image.Format.FMT_BGR888,
        fps=60,
        buff_num=3,
    )
    cam.skip_frames(WARMUP_FRAMES)
    print("[calib] camera ready\n")

    samples_by_distance = {}

    for idx, distance_mm in enumerate(CALIBRATION_DISTANCES_MM):
        print(
            "[calib] [{}/{}] Place target at {} mm ...".format(
                idx + 1, len(CALIBRATION_DISTANCES_MM), distance_mm
            )
        )
        # 给用户几秒准备时间，把靶放到指定位置
        time.sleep_ms(2000)

        samples = collect_samples(cam, distance_mm, SAMPLING_DURATION_S, disp)
        samples_by_distance[distance_mm] = samples
        print(
            "[calib]   collected {} valid frames at {} mm".format(
                len(samples), distance_mm
            )
        )

    # 释放摄像头
    del cam

    # 计算标定结果
    result = compute_calibration(samples_by_distance)
    if result is None:
        print("[calib] calibration FAILED!")
        return 1

    fx_final, fy_final, per_distance = result

    # 写入 JSON 文件
    save_camera_params(
        CALIB_FILE,
        fx=round(fx_final, 2),
        fy=round(fy_final, 2),
        cx=CAMERA_CX,
        cy=CAMERA_CY,
        dist_coeffs=DIST_COEFFS,
        extra_info={
            "calibration_date": "unknown",
            "samples": per_distance,
            "target_mm": [TARGET_W_MM, TARGET_H_MM],
        },
    )

    print("\n[calib] Done! Restart find_circle_pose.py to use calibrated values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
