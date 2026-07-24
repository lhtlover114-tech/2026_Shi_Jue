# ============================================================
# 焦距标定脚本（与 measure.py 使用相同的 HSV 白色检测逻辑）
#
# 使用方法：
#   1. 修改 CALIB_DISTANCE_MM 为 A4 纸到摄像头的实际距离（mm）
#   2. 在 MaixCAM2 上运行此脚本
#   3. 将 A4 纸正对摄像头，放在指定距离处，保持稳定
#   4. 脚本自动采集若干帧，计算 fx/fy
#   5. 把输出的 fx/fy 填到 measure.py 的 CAMERA_FX / CAMERA_FY
#
# 原理：fx = pixel_w × distance / real_w
#       fy = pixel_h × distance / real_h
# ============================================================

from maix import camera, display, image, app
from maix import time as maix_time
import cv2
import numpy as np
import math

# ===================== 标定配置 =====================
CALIB_DISTANCE_MM = 500       # A4 纸到摄像头的实际距离（mm），请根据实际情况修改
SAMPLING_TIME_S = 3.0         # 采集时长（秒）
MIN_SAMPLES = 10              # 最少有效帧数

A4_W = 250.0                  # A4 纸实际宽度 mm
A4_H = 170.0                  # A4 纸实际高度 mm

CAM_W = 640                   # 与 measure.py 保持一致
CAM_H = 480
# ===================================================

print("=" * 50)
print("  焦距标定（HSV 白色检测）")
print(f"  靶标: {A4_W:.0f} x {A4_H:.0f} mm")
print(f"  距离: {CALIB_DISTANCE_MM} mm")
print(f"  采集: {SAMPLING_TIME_S:.1f} 秒")
print("=" * 50)

cam = camera.Camera(CAM_W, CAM_H)
for _ in range(30):
    cam.read()
disp = display.Display()

print("开始采集，请保持 A4 纸稳定...")

samples_w = []
samples_h = []
start_ms = maix_time.ticks_ms()
duration_ms = int(SAMPLING_TIME_S * 1000)

while True:
    elapsed = (maix_time.ticks_ms() - start_ms) & 0xFFFFFFFF
    if elapsed >= duration_ms:
        break

    img = cam.read()
    if img is None:
        continue

    cv_img = image.image2cv(img, ensure_bgr=True, copy=False)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

    # ---- 与 measure.py 完全相同的白色检测逻辑 ----
    mask_white = cv2.inRange(hsv, np.array([0, 0, 180]),
                                   np.array([180, 40, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

    contours_w, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0
    best_bw = 0
    best_bh = 0

    for cnt in contours_w:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)

        # 宽高比检查
        aspect = bw / bh if bw < bh else bh / bw
        a4_aspect = A4_W / A4_H
        if abs(aspect - a4_aspect) / a4_aspect > 0.25:
            continue

        # 取面积最大的
        if area > best_area:
            best_area = area
            best = (bx, by, bw, bh)
            best_bw = bw
            best_bh = bh

    # 在图像上绘制检测结果
    if best is not None:
        bx, by, bw, bh = best
        samples_w.append(bw)
        samples_h.append(bh)
        img.draw_rect(bx, by, bw, bh, image.Color.from_rgb(0, 255, 0), 2)

    # 屏幕叠加信息
    remaining_s = max(0.0, (duration_ms - elapsed) / 1000.0)
    img.draw_string(4, 4,
                    f"CALIB D={CALIB_DISTANCE_MM}mm",
                    image.Color.from_rgb(0, 255, 0), scale=1.2, thickness=2)
    img.draw_string(4, 26,
                    f"samples:{len(samples_w)}  remain:{remaining_s:.1f}s",
                    image.Color.from_rgb(255, 255, 0), scale=1.2, thickness=2)
    if best is not None:
        img.draw_string(4, 48,
                        f"W:{best_bw:.0f}  H:{best_bh:.0f} px",
                        image.Color.from_rgb(0, 255, 0), scale=1.2, thickness=2)

    disp.show(img)

# ===================== 计算焦距 =====================
print("\n" + "=" * 50)
print(f"  采集完成: {len(samples_w)} 帧")
print("=" * 50)

if len(samples_w) < MIN_SAMPLES:
    print(f"[ERROR] 有效帧数不足 ({len(samples_w)} < {MIN_SAMPLES})，请重新标定")
    print("[提示] 检查 A4 纸是否在画面中、光线是否充足")
    exit(1)

# 取中位数抗干扰
sorted_w = sorted(samples_w)
sorted_h = sorted(samples_h)
median_w = sorted_w[len(sorted_w) // 2]
median_h = sorted_h[len(sorted_h) // 2]

fx = median_w * CALIB_DISTANCE_MM / A4_W
fy = median_h * CALIB_DISTANCE_MM / A4_H

# 计算标准差，评估采集稳定性
if len(samples_w) >= 3:
    mean_w = sum(samples_w) / len(samples_w)
    mean_h = sum(samples_h) / len(samples_h)
    std_w = math.sqrt(sum((s - mean_w) ** 2 for s in samples_w) / len(samples_w))
    std_h = math.sqrt(sum((s - mean_h) ** 2 for s in samples_h) / len(samples_h))
    cv_w = std_w / mean_w * 100
    cv_h = std_h / mean_h * 100
    print(f"  像素宽度: 中位数={median_w:.1f}  均值={mean_w:.1f}  std={std_w:.1f}  CV={cv_w:.1f}%")
    print(f"  像素高度: 中位数={median_h:.1f}  均值={mean_h:.1f}  std={std_h:.1f}  CV={cv_h:.1f}%")
    if cv_w > 5.0 or cv_h > 5.0:
        print(f"  [WARNING] 波动较大（CV>{5.0}%），建议保持靶标稳定后重新标定")

print()
print("=" * 50)
print("  标定结果 —— 复制以下值到 measure.py")
print("=" * 50)
print(f"  CAMERA_FX = {fx:.2f}")
print(f"  CAMERA_FY = {fy:.2f}")
print("=" * 50)

# 验算：用标出的焦距反算距离，应该接近标定距离
check_d_w = (A4_W * fx) / median_w
check_d_h = (A4_H * fy) / median_h
print(f"  验算距离: D_w={check_d_w:.0f}mm  D_h={check_d_h:.0f}mm")
print(f"  误差: {abs(check_d_w - CALIB_DISTANCE_MM):.0f}mm / {abs(check_d_h - CALIB_DISTANCE_MM):.0f}mm")
print("=" * 50)
