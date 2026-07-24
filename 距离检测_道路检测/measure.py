# ============================================================
# 电赛 C 题 - 两步法测距（标定焦距版）
# 第一步：HSV过滤白色 → 按尺寸比例筛选A4纸
# 第二步：在A4纸区域内 → 找黑色图形 → 算尺寸
#
# 焦距来源：运行 calibrate_focal.py 标定后，将输出的 fx/fy
#          填入下方 CAMERA_FX / CAMERA_FY。
# ============================================================

from maix import camera, display, image, app
from maix import time as maix_time
import cv2
import numpy as np
import math

# ===================== 配置 =====================
A4_W = 170.0         # A4纸实际宽度 mm
A4_H = 250.0         # A4纸实际高度 mm

# ---- 标定焦距（运行 calibrate_camera.py 后更新）----
CAMERA_FX = 358.0    # x 方向焦距，对应宽度方向
CAMERA_FY = 800.0    # y 方向焦距，对应高度方向

# ---- 摄像头分辨率（需与标定时一致）----
CAM_W = 640
CAM_H = 480
# ===============================================

print("=== 两步法测距（标定焦距版）===")
cam = camera.Camera(CAM_W, CAM_H)
for _ in range(30):
    cam.read()
disp = display.Display()
print("开始测量...")

frame_id = 0

while not app.need_exit():
    frame_id += 1
    if frame_id % 2 == 0:
        continue

    img = cam.read()
    if img is None:
        continue
    t0 = maix_time.ticks_ms()

    # ===== 第一步：HSV过滤白色 → 找A4纸 =====
    cv_img = image.image2cv(img, ensure_bgr=True, copy=False)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(hsv, np.array([0, 0, 180]),
                                   np.array([180, 40, 255]))
    kernel = np.ones((5, 5), np.uint8)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
    contours_w, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    D = 0.0
    x = 0.0
    found = False

    for cnt in contours_w:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)

        # ---- 用A4纸物理尺寸 + 标定焦距 筛掉假A4纸 ----

        # 1. 检查宽高比是否接近A4纸
        #    允许±25%误差
        aspect = bw / bh if bw < bh else bh / bw
        a4_aspect = A4_W / A4_H
        if abs(aspect - a4_aspect) / a4_aspect > 0.25:
            continue

        # 2. 用标定焦距分别按宽度和高度估算距离，取平均
        if CAMERA_FX > 0 and CAMERA_FY > 0:
            D_w = (A4_W * CAMERA_FX) / bw
            D_h = (A4_H * CAMERA_FY) / bh
            D_test = (D_w + D_h) * 0.5
        else:
            continue

        # 3. 检查距离是否在合理范围（80~220cm）
        if D_test < 800 or D_test > 2200:
            continue

        # 4. 用算出的D反推像素宽/高，检查是否匹配（误差<30%）
        expected_bw = (A4_W * CAMERA_FX) / D_test
        expected_bh = (A4_H * CAMERA_FY) / D_test
        if abs(bw - expected_bw) / expected_bw > 0.3:
            continue
        if abs(bh - expected_bh) / expected_bh > 0.3:
            continue

        # ---- 通过筛选，确认是A4纸 ----
        D = D_test

        img.draw_rect(bx, by, bw, bh,
                      image.Color.from_rgb(0, 255, 0), 2)
        img.draw_string(bx, by - 18,
                        f"D={D:.0f}mm",
                        image.Color.from_rgb(0, 255, 0))
        print(f"[A4纸] w={bw}px h={bh}px D_w={D_w:.0f} D_h={D_h:.0f} D={D:.0f}mm")

        # ===== 第二步：在A4纸区域内找黑色图形 =====
        roi = cv_img[by:by + bh, bx:bx + bw]
        if roi.size == 0:
            continue

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi_bin = cv2.threshold(roi_gray, 100, 255,
                                   cv2.THRESH_BINARY_INV)
        contours_s, _ = cv2.findContours(roi_bin, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

        for sc in contours_s:
            s_area = cv2.contourArea(sc)
            if s_area < 200:
                continue
            s_peri = cv2.arcLength(sc, True)
            s_approx = cv2.approxPolyDP(sc, 0.02 * s_peri, True)
            if len(s_approx) not in (3, 4):
                continue

            sx, sy, sw, sh = cv2.boundingRect(sc)
            sx_orig = bx + sx
            sy_orig = by + sy

            # 用标定焦距 + 距离 反算图形实际宽度
            if CAMERA_FX > 0 and D > 0:
                x = (sw * D) / CAMERA_FX

            img.draw_rect(sx_orig, sy_orig, sw, sh,
                          image.Color.from_rgb(255, 0, 0), 2)
            shape_name = "tri" if len(s_approx) == 3 else "rect"
            img.draw_string(sx_orig, sy_orig - 15,
                            f"{shape_name} x={x:.1f}mm",
                            image.Color.from_rgb(255, 0, 0))

        found = True
        break

    if not found:
        img.draw_string(10, 20, "No A4 paper",
                        image.Color.from_rgb(255, 0, 0))

    elapsed = maix_time.ticks_ms() - t0
    if elapsed < 50:
        maix_time.sleep_ms(50 - elapsed)

    disp.show(img)
