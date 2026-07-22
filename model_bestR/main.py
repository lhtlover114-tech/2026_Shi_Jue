# -*- coding: utf-8 -*-
"""
MaixCAM2 单类 Kuang 检测实机运行脚本 (电赛 E 题自瞄) - 带耗时统计

模型: best_npu.axmodel (单类 Kuang, INT8, 640x480, NPU2)
配套: best.mud (labels = Kuang)

耗时统计 (MaixPy time API):
  - time.ticks_ms()    : 从开机起的毫秒数 (uint64, 单调递增)
  - time.ticks_diff(last, now) : 计算 now - last 的时间差 (ms)
  - time.fps()         : 官方 FPS 计算 (最近 20 次调用的平均)

各阶段耗时:
  - cap : 摄像头采集
  - det : 检测 (NPU推理 + 后处理DFL/NMS)  ← 核心指标
  - draw: 绘制框/十字/文字
  - show: 显示刷新
  - tot : 一帧总耗时, 1000/tot = 实际FPS
"""
from maix import camera, display, image, nn, app, time

# ==================== 配置 ====================
MODEL_PATH = "/root/models/best.mud"

CONF_TH = 0.5
IOU_TH  = 0.45

DRAW_BOX   = True
DRAW_CROSS = True
DRAW_COORD = True
SHOW_TIMING = True     # 屏幕显示各阶段耗时
DRAW_DEBUG = True      # 串口打印调试

COLOR_BOX   = image.COLOR_RED
COLOR_CROSS = image.COLOR_YELLOW
COLOR_TEXT  = image.COLOR_WHITE
COLOR_FPS   = image.COLOR_GREEN
COLOR_TIME  = image.COLOR_BLUE


# ==================== 初始化 ====================
print(f"[+] loading model: {MODEL_PATH}")
detector = nn.YOLO11(model=MODEL_PATH, dual_buff=True)

CAM_W = detector.input_width()
CAM_H = detector.input_height()
CAM_FMT = detector.input_format()
print(f"[+] model expects: {CAM_W}x{CAM_H} {CAM_FMT}")
print(f"[+] labels      : {detector.labels}")

cam = camera.Camera(CAM_W, CAM_H, CAM_FMT)
disp = display.Display()
cam.skip_frames(30)
print("[+] start detection...")


# ==================== 主循环 ====================
def smooth(old, new, alpha=0.8):
    """指数平滑, alpha 越大越平滑"""
    return old * alpha + new * (1 - alpha) if old > 0 else new

sm_cap = sm_det = sm_draw = sm_show = sm_tot = 0.0
frame_cnt = 0
DEBUG_EVERY = 30

# FPS 起始点
time.fps_start()

while not app.need_exit():
    t0 = time.ticks_ms()

    # === 1. 采集 ===
    img = cam.read()
    t1 = time.ticks_ms()

    # === 2. 检测 (NPU 推理 + 后处理) ===
    objs = detector.detect(img, conf_th=CONF_TH, iou_th=IOU_TH)
    t2 = time.ticks_ms()

    # 调试打印
    if DRAW_DEBUG and frame_cnt % DEBUG_EVERY == 0:
        scores = [f"{o.score:.2f}" for o in objs]
        print(f"[dbg] frame={frame_cnt} objs={len(objs)} scores={scores}")

    # === 3. 绘制 ===
    kuang_center = None
    for obj in objs:
        cx = obj.x + obj.w // 2
        cy = obj.y + obj.h // 2
        kuang_center = (cx, cy)
        if DRAW_BOX:
            img.draw_rect(obj.x, obj.y, obj.w, obj.h, color=COLOR_BOX, thickness=2)
            img.draw_string(obj.x, obj.y - 15, f"Kuang:{obj.score:.2f}",
                            color=COLOR_BOX, scale=1.2)
        if DRAW_CROSS:
            img.draw_line(cx - 10, cy, cx + 10, cy, color=COLOR_CROSS, thickness=2)
            img.draw_line(cx, cy - 10, cx, cy + 10, color=COLOR_CROSS, thickness=2)

    if DRAW_COORD:
        if kuang_center:
            cx, cy = kuang_center
            img.draw_string(10, 10, f"center: ({cx}, {cy})",
                            color=COLOR_TEXT, scale=1.2)
        else:
            img.draw_string(10, 10, "no target", color=image.COLOR_RED, scale=1.2)
    t3 = time.ticks_ms()

    # === 耗时计算 (MaixPy: ticks_diff(last, now) = now - last) ===
    cap_ms  = time.ticks_diff(t0, t1)   # t1 - t0
    det_ms  = time.ticks_diff(t1, t2)   # t2 - t1
    draw_ms = time.ticks_diff(t2, t3)   # t3 - t2

    sm_cap  = smooth(sm_cap, cap_ms)
    sm_det  = smooth(sm_det, det_ms)
    sm_draw = smooth(sm_draw, draw_ms)

    # FPS 显示 (官方 API)
    fps = time.fps()
    img.draw_string(10, 35, f"FPS: {fps:.1f} objs: {len(objs)}",
                    color=COLOR_FPS, scale=1.2)

    # 耗时统计显示
    if SHOW_TIMING:
        y = 60
        img.draw_string(10, y, f"cap:  {sm_cap:5.1f} ms", color=COLOR_TIME, scale=1.0); y += 18
        img.draw_string(10, y, f"det:  {sm_det:5.1f} ms", color=COLOR_TIME, scale=1.0); y += 18
        img.draw_string(10, y, f"draw: {sm_draw:5.1f} ms", color=COLOR_TIME, scale=1.0); y += 18

    # === 4. 显示 ===
    disp.show(img)
    t4 = time.ticks_ms()

    sm_show = smooth(sm_show, time.ticks_diff(t3, t4))   # t4 - t3
    sm_tot  = smooth(sm_tot, time.ticks_diff(t0, t4))    # t4 - t0

    # 串口打印 (含总耗时, 方便看真实帧率)
    if DRAW_DEBUG and frame_cnt % DEBUG_EVERY == 0:
        print(f"[time] cap={sm_cap:.1f} det={sm_det:.1f} draw={sm_draw:.1f} "
              f"show={sm_show:.1f} tot={sm_tot:.1f}ms  fps={1000/sm_tot:.1f}")

    frame_cnt += 1

print("[+] exited")
