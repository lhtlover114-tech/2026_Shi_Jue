"""基于 移植K230.py 的相机标定脚本（纯 OpenCV 检测，无 YOLO）。

原理 —— 小孔成像模型：fx = 像素宽度 × 距离 / 靶实际宽度

使用方法：
  1. 修改下方 CALIBRATION_DISTANCES_MM 为你要测量的实际距离（mm）
  2. 在 MaixCAM2 上运行此脚本
  3. 按照屏幕提示，把 29.7cm×21cm 靶依次放在每个距离处
  4. 每个距离保持稳定，脚本自动采集若干帧后跳到下一个距离
  5. 运行结束后复制输出的 fx/fy，手动填到 find_circle_pose.py 中
"""

import math
import cv2
import numpy as np

from maix import camera, display, image, time

# 复用 移植K230.py 的检测函数和常量
from 移植K230 import (
    detect_rectangle,
    scale_points,
    CAM_W,
    CAM_H,
    PROC_W,
    PROC_H,
)

# ============================================================
# 靶尺寸 & 标定配置
# ============================================================

# 29.7 cm x 21 cm A4 横向矩形靶（单位 mm）
TARGET_W_MM = 297.0
TARGET_H_MM = 210.0

# 已知距离列表 (mm)，把靶依次放在这些距离处
CALIBRATION_DISTANCES_MM = [250, 500, 750, 1000, 1250]

# 每个距离的采集时长 (秒)
SAMPLING_DURATION_S = 3.0

# 每个距离最少需要的有效检测帧数
MIN_SAMPLES_PER_DISTANCE = 10

# 摄像头预热帧数
WARMUP_FRAMES = 20


# ============================================================
# 辅助函数（移植K230.py 中没有，这里补充）
# ============================================================

def distance_2d(first, second):
    """两点欧氏距离。"""
    dx = float(second[0]) - float(first[0])
    dy = float(second[1]) - float(first[1])
    return math.sqrt(dx * dx + dy * dy)


def order_quad_points(points):
    """把任意四角点排序为：左上、右上、右下、左下。"""
    if points is None or len(points) != 4:
        return None

    pts = np.array(points, dtype=np.float32)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmax(diffs)]
    bottom_left = pts[np.argmin(diffs)]

    return np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32,
    )


def target_pixel_width(ordered_points):
    """横向靶宽在图像中的平均像素宽度：(上边 + 下边) / 2。"""
    if ordered_points is None or len(ordered_points) != 4:
        return 0.0
    top_width = distance_2d(ordered_points[0], ordered_points[1])
    bottom_width = distance_2d(ordered_points[3], ordered_points[2])
    return (top_width + bottom_width) * 0.5


def compute_pixel_height(ordered_points):
    """靶在图像中的平均像素高度：(左边 + 右边) / 2。"""
    if ordered_points is None or len(ordered_points) != 4:
        return 0.0
    left_height = distance_2d(ordered_points[0], ordered_points[3])
    right_height = distance_2d(ordered_points[1], ordered_points[2])
    return (left_height + right_height) * 0.5


# ============================================================
# 标定核心逻辑
# ============================================================

def collect_samples(cam, target_distance_mm, duration_s, disp):
    """在指定距离处采集若干 (像素宽度, 像素高度) 样本。"""
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
            # 画出检测到的矩形框
            for i in range(4):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % 4]
                img.draw_line(x1, y1, x2, y2, image.COLOR_GREEN, thickness=2)

            ordered = order_quad_points(points)
            if ordered is not None:
                pw = target_pixel_width(ordered)
                ph = compute_pixel_height(ordered)
                if pw > 1.0 and ph > 1.0:
                    samples.append((pw, ph))

        # 屏幕状态叠加
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
    """根据各距离采集的样本计算 fx / fy。"""
    print("\n" + "=" * 64)
    print("  {:>6s}  {:>7s}  {:>8s}  {:>8s}  {:>7s}  {:>7s}".format(
        "Dist", "Samples", "Med_W", "Med_H", "fx", "fy"))
    print("-" * 64)

    fx_values = []
    fy_values = []

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

    if len(fx_values) >= 2:
        fx_range = max(fx_values) - min(fx_values)
        fx_variation = fx_range / fx_final * 100.0
        print("  fx variation across distances: {:.1f}%".format(fx_variation))
        if fx_variation > 5.0:
            print("  ** WARNING: fx varies >5%, re-check measurements! **")

    return fx_final, fy_final


# ============================================================
# 主入口
# ============================================================

def main():
    print("=" * 64)
    print("  Camera Calibration  --  Pure OpenCV (移植K230)")
    print("  Target: {:.0f} x {:.0f} mm".format(TARGET_W_MM, TARGET_H_MM))
    print("  Distances: {} mm".format(CALIBRATION_DISTANCES_MM))
    print("  Sampling: {:.1f}s per distance".format(SAMPLING_DURATION_S))
    print("=" * 64)

    disp = display.Display()

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
        time.sleep_ms(2000)

        samples = collect_samples(cam, distance_mm, SAMPLING_DURATION_S, disp)
        samples_by_distance[distance_mm] = samples
        print(
            "[calib]   collected {} valid frames at {} mm".format(
                len(samples), distance_mm
            )
        )

    del cam

    result = compute_calibration(samples_by_distance)
    if result is None:
        print("[calib] calibration FAILED!")
        return 1

    fx_final, fy_final = result

    print("\n" + "=" * 64)
    print("  CALIBRATION DONE")
    print("  Copy these values into find_circle_pose.py:")
    print("=" * 64)
    print("  CAMERA_FX = {:.2f}".format(fx_final))
    print("  CAMERA_FY = {:.2f}".format(fy_final))
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
